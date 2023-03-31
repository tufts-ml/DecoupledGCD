import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gcd_cluster.ss_gmm import SSGMM

from gcd_data.get_datasets import get_class_splits, get_datasets

from ccgaussian.augment import sim_gcd_train, sim_gcd_test
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
    parser.add_argument("--var_milestone", type=int, default=25)
    # loss hyperparameters
    parser.add_argument("--w_nll", type=float, default=1e-2,
                        help="Negative log-likelihood weight for embedding network")
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
    novel_embeds = torch.empty((0, model.embed_len))
    normal_embeds = torch.empty((0, model.embed_len))
    normal_targets = torch.Tensor([])
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
    return SSGMM(normal_embeds.cpu().numpy(),
                 normal_targets.cpu().numpy(),
                 novel_embeds.cpu().numpy(),
                 model.num_classes)


def train_gcd(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # init dataloaders
    train_loader, valid_loader, test_loader, args = get_gcd_dataloaders(args)
    # normal classes after target transform according to GCDdatasets API
    normal_classes = torch.arange(args.num_labeled_classes).to(device)
    # init model
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    model = DinoCCG(num_classes, args.init_var, args.end_var, args.var_milestone).to(device)
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
    phases = ["train", "valid", "test"]
    # init loss
    loss_func = NDCCFixedSoftLoss(args.w_nll)
    # init tensorboard, with random comment to stop overlapping runs
    writer = SummaryWriter(args.label, comment=str(random.randint(0, 9999)))
    # metric dict for recording hparam metrics
    metric_dict = {}
    # initialze GMM
    gmm = init_gmm(model, train_loader, device)
    # model training
    for epoch in range(args.num_epochs):
        # update model variance
        model.anneal_var(epoch)
        # Each epoch has a training, validation, and test phase
        for phase in phases:
            if phase == "train":
                model.train()
                dataloader = train_loader
            elif phase == "valid":
                model.eval()
                dataloader = valid_loader
            else:
                model.eval()
                dataloader = test_loader
            # vars for tensorboard stats
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            epoch_nll = 0.
            epoch_sigma2s = 0.
            for batch in dataloader:
                # handle differing dataset formats
                if phase == "train":
                    data, targets, uq_idxs, norm_mask = batch
                else:
                    data, targets, uq_idxs = batch
                    norm_mask = torch.isin(targets, normal_classes)
                # move data to device
                data = data.to(device)
                targets = targets.long().to(device)
                norm_mask = norm_mask.to(device)
                # create normal soft targets
                num_samples = data.shape[0]
                soft_targets = torch.zeros((num_samples, num_classes)).to(device)
                soft_targets[torch.arange(num_samples)[norm_mask], targets[norm_mask]] = 1
                # forward and loss
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits, embeds, means, sigma2s = model(data)
                    # create novel soft targets
                    novel_embeds = embeds.detach().cpu().numpy()
                    soft_targets[torch.arange(num_samples)[~norm_mask]] = \
                        torch.Tensor(gmm._e_step(novel_embeds)[1]).to(device)
                    loss = loss_func(logits, embeds, means, sigma2s, soft_targets)
                # backward and optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optim.step()
                    scheduler.step()
                # calculate statistics, masking novel classes
                _, preds = torch.max(logits[norm_mask], 1)
                epoch_loss = (loss.item() * data.size(0) +
                              cnt * epoch_loss) / (cnt + data.size(0))
                if len(preds) > 0:
                    epoch_acc = (torch.sum(preds == targets[norm_mask].data) +
                                 epoch_acc * cnt).double() / (cnt + len(preds))
                    epoch_nll = (NDCCFixedSoftLoss.nll_loss(
                        embeds[norm_mask], means, sigma2s, targets[norm_mask]) +
                        epoch_nll * cnt) / (cnt + len(preds))
                epoch_sigma2s = (torch.mean(sigma2s) * data.size(0) +
                                 epoch_sigma2s * cnt) / (cnt + data.size(0))
                cnt += data.size(0)
            # TODO update SSGMM using classifier predictions for unlabeled data
            # get phase label
            if phase == "train":
                phase_label = "Train"
            elif phase == "valid":
                phase_label = "Valid"
            else:
                phase_label = "Test"
            # output statistics
            writer.add_scalar(f"{phase_label}/Average Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase_label}/Average Accuracy", epoch_acc, epoch)
            writer.add_scalar(f"{phase_label}/Average NLL", epoch_nll, epoch)
            # only output variance for training since other phases will match it
            if phase == "train":
                writer.add_scalar(f"{phase_label}/Average Variance Mean", epoch_sigma2s, epoch)
            if phase != "train":
                # record end of training stats, grouped as Metrics in Tensorboard
                if epoch == args.num_epochs - 1:
                    # note non-numeric values (NaN, None, ect.) will cause entry
                    # to not be displayed in Tensorboard HPARAMS tab
                    metric_dict.update({
                        f"Metrics/{phase_label}_loss": epoch_loss,
                        f"Metrics/{phase_label}_accuracy": epoch_acc,
                        f"Metrics/{phase_label}_nll": epoch_nll,
                    })
    # record hparams all at once and after all other writer calls
    # to avoid issues with Tensorboard changing output file
    writer.add_hparams({
        "lr_e": args.lr_e,
        "lr_c": args.lr_c,
        "init_var": args.init_var,
        "end_var": args.end_var,
        "w_nll": args.w_nll,
    }, metric_dict)
    torch.save(model.state_dict(), Path(writer.get_logdir()) / f"{args.num_epochs}.pt")


if __name__ == "__main__":
    args = get_args()
    train_gcd(args)
