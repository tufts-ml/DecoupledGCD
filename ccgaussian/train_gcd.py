import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from gcd_cluster.ss_gmm import DeepSSGMM

from gcd_data.get_datasets import get_class_splits, get_datasets

from ccgaussian.augment import sim_gcd_train, sim_gcd_test
from ccgaussian.logger import AverageWriter
from ccgaussian.loss import NDCCFixedSoftLoss
from ccgaussian.model import DinoCCG
from ccgaussian.scheduler import warm_cos_scheduler


def get_args():
    parser = argparse.ArgumentParser()
    # dataset and split arguments
    parser.add_argument(
        "--dataset_name", type=str, default="cub",
        choices=["NovelCraft", "cifar10", "cifar100", "imagenet_100", "cub", "scars",
                 "fgvc_aricraft", "herbarium_19"],
        help="options: NovelCraft, cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, " +
             "herbarium_19")
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    # model label for logging
    parser.add_argument("--label", type=str, default=None)
    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr_e", type=float, default=1e-5,
                        help="Learning rate for embedding v(x)")
    parser.add_argument("--lr_c", type=float, default=1e-3,
                        help="Learning rate for linear classifier {w_y, b_y}")
    parser.add_argument("--init_var", type=float, default=1,
                        help="Initial variance")
    parser.add_argument("--end_var", type=float, default=3e-1,
                        help="Final variance")
    parser.add_argument("--var_warmup", type=int, default=25)
    # loss hyperparameters
    parser.add_argument("--w_nll", type=float, default=1e-2,
                        help="Negative log-likelihood weight for embedding network")
    parser.add_argument("--w_novel", type=float, default=1,
                        help="Novel loss weight")
    parser.add_argument("--novel_warmup", type=float, default=25,
                        help="Novel loss weight warmup epochs")
    args = parser.parse_args()
    # prepend runs folder to label if given
    if args.label is not None:
        args.label = "runs/" + args.label
    # adjust learning rates based on batch size
    args.lr_e *= args.batch_size / 256
    args.lr_c *= args.batch_size / 256
    return args


def get_gcd_dataloaders(args):
    """Get novelty detection DataLoaders

    Args:
        args (Namespace): args containing dataset and prop_train_labels

    Returns:
        tuple: train_loader: Normal training set
            valid_loader: Normal and novel validation set
            test_loader: Normal and novel test set
            args: args updated with num_labeled_classes and num_unlabeled_classes
    """
    train_trans = sim_gcd_train()
    test_trans = sim_gcd_test()
    args = get_class_splits(args)
    train_dataset, valid_dataset, test_dataset = get_datasets(
        args.dataset_name, train_trans, test_trans, args)[:3]
    # add number of labeled and unlabeled classes to args
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    # construct DataLoaders
    # need to set num_workers=0 in Windows due to torch.multiprocessing pickling limitation
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "shuffle": True,
    }
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    valid_loader = DataLoader(valid_dataset, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **dataloader_kwargs)
    return train_loader, valid_loader, test_loader, args


def init_gmm(model: DinoCCG, train_loader, device):
    novel_embeds = torch.empty((0, model.embed_len)).to(device)
    normal_embeds = torch.empty((0, model.embed_len)).to(device)
    normal_targets = torch.tensor([], dtype=int).to(device)
    for batch in train_loader:
        data, targets, _, norm_mask = batch
        # move data to device
        data = data.to(device)
        targets = targets.long().to(device)
        norm_mask = norm_mask.to(device)
        with torch.set_grad_enabled(False):
            _, embeds, _, _ = model(data)
            novel_embeds = torch.vstack((novel_embeds, embeds[~norm_mask]))
            normal_embeds = torch.vstack((normal_embeds, embeds[norm_mask]))
            normal_targets = torch.hstack((normal_targets, targets[norm_mask]))
    return DeepSSGMM(normal_embeds.cpu().numpy(),
                     normal_targets.cpu().numpy(),
                     novel_embeds.cpu().numpy(),
                     model.num_classes,
                     model.init_var)


def train_gcd(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # init dataloaders
    train_loader, valid_loader, test_loader, args = get_gcd_dataloaders(args)
    # normal classes after target transform according to GCDdatasets API
    normal_classes = torch.arange(args.num_labeled_classes).to(device)
    # init model
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    model = DinoCCG(num_classes, args.init_var, args.end_var, args.var_warmup).to(device)
    # init optimizer
    optim = torch.optim.AdamW([
        {
            "params": model.dino.parameters(),
            "lr": args.lr_e,
            "weignt_decay": 5e-4,
        },
        {
            "params": model.classifier.parameters(),
            "lr": args.lr_c,
        },
    ])
    scheduler = warm_cos_scheduler(optim, args.num_epochs, train_loader)
    phases = ["Train", "Valid", "Test"]
    # init loss
    loss_func = NDCCFixedSoftLoss(args.w_nll, args.w_novel)
    # init tensorboard, with random comment to stop overlapping runs
    av_writer = AverageWriter(args.label, comment=str(random.randint(0, 9999)))
    # metric dict for recording hparam metrics
    metric_dict = {}
    # initialze GMM
    gmm = init_gmm(model, train_loader, device)
    # model training
    for epoch in range(args.num_epochs):
        # update model variance and loss w_novel
        model.anneal_var(epoch)
        loss_func.anneal_w_novel(epoch)
        # each epoch has a training, validation, and test phase
        for phase in phases:
            # get dataloader
            if phase == "Train":
                model.train()
                dataloader = train_loader
            elif phase == "Valid":
                model.eval()
                dataloader = valid_loader
            else:
                model.eval()
                dataloader = test_loader
            # vars for m step
            if phase == "Train":
                novel_embeds = torch.empty((0, model.embed_len)).to(device)
                novel_resp = torch.empty((0, num_classes)).to(device)
                normal_embeds = torch.empty((0, model.embed_len)).to(device)
                normal_targets = torch.tensor([], dtype=int).to(device)
            for batch in dataloader:
                # handle differing dataset formats
                if phase == "Train":
                    data, targets, uq_idxs, norm_mask = batch
                else:
                    data, targets, uq_idxs = batch
                    norm_mask = torch.isin(targets.to(device), normal_classes)
                # move data to device
                data = data.to(device)
                targets = targets.long().to(device)
                norm_mask = norm_mask.to(device)
                # create normal soft targets
                num_samples = data.shape[0]
                soft_targets = torch.zeros((num_samples, num_classes)).to(device)
                soft_targets[torch.arange(num_samples).to(device)[norm_mask],
                             targets[norm_mask]] = 1
                # forward and loss
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "Train"):
                    logits, embeds, means, sigma2s = model(data)
                    # create novel soft targets
                    soft_targets[~norm_mask] = torch.Tensor(
                        gmm.deep_e_step(embeds[~norm_mask].detach().cpu().numpy())).to(device)
                    loss = loss_func(logits, embeds, means, sigma2s, soft_targets, norm_mask)
                # backward and optimize only if in training phase
                if phase == "Train":
                    loss.backward()
                    optim.step()
                    scheduler.step()
                # calculate statistics, masking novel classes
                _, preds = torch.max(logits[norm_mask], 1)
                if len(preds) > 0:
                    av_writer.update(f"{phase}/Average Accuracy",
                                     torch.mean(preds == targets[norm_mask].data),
                                     len(preds))
                    av_writer.update(
                        f"{phase}/Average NLL",
                        NDCCFixedSoftLoss.nll_loss(
                            embeds[norm_mask], means, sigma2s, targets[norm_mask]) / len(preds),
                        len(preds))
                av_writer.update(f"{phase}/Average Loss",
                                 loss.item(), num_samples)
                # track mean pseudo-label confidence
                av_writer.update(f"{phase}/Average Pseudo-label Confidence",
                                 torch.max(soft_targets[~norm_mask], dim=1)[0].mean(),
                                 (~norm_mask).sum())
                # only output annealed values for training since other phases will match it
                if phase == "Train":
                    av_writer.update(f"{phase}/Average Variance Mean",
                                     torch.sum(sigma2s), num_samples)
                    av_writer.update(f"{phase}/Novel Weight",
                                     loss_func.an_w_novel, num_samples)
                # cache data for m step
                if phase == "Train":
                    novel_embeds = torch.vstack((novel_embeds, embeds[~norm_mask]))
                    novel_resp = torch.vstack((novel_resp, soft_targets[~norm_mask]))
                    normal_embeds = torch.vstack((normal_embeds, embeds[norm_mask]))
                    normal_targets = torch.hstack((normal_targets, targets[norm_mask]))
            # update SSGMM using classifier predictions for unlabeled data
            if phase == "Train":
                gmm.deep_m_step(
                    normal_embeds.detach().cpu().numpy(),
                    normal_targets.detach().cpu().numpy(),
                    novel_embeds.detach().cpu().numpy(),
                    novel_resp.detach().cpu().numpy(),
                    float(model.sigma2s[0].detach().cpu()))
            # record end of training stats, grouped as Metrics in Tensorboard
            if phase != "Train" and epoch == args.num_epochs - 1:
                # note non-numeric values (NaN, None, ect.) will cause entry
                # to not be displayed in Tensorboard HPARAMS tab
                metric_dict.update({
                    f"Metrics/{phase}_loss": av_writer.scalars[f"{phase}/Average Loss"],
                    f"Metrics/{phase}_accuracy": av_writer.scalars[f"{phase}/Average Accuracy"],
                    f"Metrics/{phase}_nll": av_writer.scalars[f"{phase}/Average NLL"],
                })
            # output statistics
            av_writer.write(epoch)
    # record hparams all at once and after all other writer calls
    # to avoid issues with Tensorboard changing output file
    av_writer.writer.add_hparams({
        "lr_e": args.lr_e,
        "lr_c": args.lr_c,
        "init_var": args.init_var,
        "end_var": args.end_var,
        "w_nll": args.w_nll,
        "w_novel": args.w_novel,
    }, metric_dict)
    torch.save(model.state_dict(), Path(av_writer.writer.get_logdir()) / f"{args.num_epochs}.pt")


if __name__ == "__main__":
    args = get_args()
    train_gcd(args)
