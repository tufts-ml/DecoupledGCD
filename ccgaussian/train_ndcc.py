import argparse
from pathlib import Path
import random

from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gcd_data.get_datasets import get_class_splits, get_datasets

from ccgaussian.dino_trans import sim_gcd_train, sim_gcd_test
from ccgaussian.loss import NDCCFixedLoss, novelty_md
from ccgaussian.model import DinoCCG


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_e", type=float, default=1e-5,
                        help="Learning rate for embedding v(x)")
    parser.add_argument("--lr_c", type=float, default=1e-3,
                        help="Learning rate for linear classifier {w_y, b_y}")
    parser.add_argument("--init_var", type=float, default=1,
                        help="Initial variance")
    parser.add_argument("--end_var", type=float, default=3e-1,
                        help="Final variance")
    parser.add_argument("--var_milestone", default=25)
    parser.add_argument("--lr_milestones", default=[30, 40, 45])
    # loss hyperparameters
    parser.add_argument("--w_nll", type=float, default=1e-2,
                        help="Negative log-likelihood weight for embedding network")
    args = parser.parse_args()
    # prepend runs folder to label if given
    if args.label is not None:
        args.label = "runs/" + args.label
    return args


def get_nd_dataloaders(args):
    """Get novelty detection DataLoaders

    Args:
        args (Namespace): args containing dataset and prop_train_labels

    Returns:
        tuple: train_loader: Normal training set
            valid_loader: Normal and novel validation set
            test_loader: Normal and novel test set
            args: args updated with num_labeled_classes and num_unlabeled_classes
    """
    train_trans = sim_gcd_test()  # TODO fix, only using test for debugging
    test_trans = sim_gcd_test()
    args = get_class_splits(args)
    dataset_dict = get_datasets(args.dataset_name, train_trans, test_trans, args)[-1]
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
    train_loader = DataLoader(dataset_dict["train_labeled"], **dataloader_kwargs)
    valid_loader = DataLoader(dataset_dict["test"], **dataloader_kwargs)
    test_loader = DataLoader(dataset_dict["train_unlabeled"], **dataloader_kwargs)
    return train_loader, valid_loader, test_loader, args


def train_ndcc(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # init dataloaders
    train_loader, valid_loader, test_loader, args = get_nd_dataloaders(args)
    normal_classes = torch.tensor(list(args.train_classes)).to(device)
    # init model
    model = DinoCCG(
        args.num_labeled_classes, args.init_var, args.end_var, args.var_milestone).to(device)
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=args.lr_milestones, gamma=0.5)
    phases = ["train", "valid", "test"]
    # init loss
    loss_func = NDCCFixedLoss(args.w_nll)
    # init tensorboard, with random comment to stop overlapping runs
    writer = SummaryWriter(args.label, comment=str(random.randint(0, 9999)))
    # metric dict for recording hparam metrics
    metric_dict = {}
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
            # novelty prediction storage for non-training phases
            novel_scores = torch.Tensor([])
            novel_labels = torch.Tensor([])
            # vars for tensorboard stats
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            epoch_nll = 0.
            epoch_sigma2s = 0.
            for data, targets, uq_idxs in dataloader:
                # forward and loss
                data = data.to(device)
                targets = targets.long().to(device)
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits, norm_embeds, means, sigma2s = model(data)
                    if phase == "train":
                        # all true mask if training
                        norm_mask = torch.ones((data.size(0),), dtype=torch.bool).to(device)
                    else:
                        # filter out novel examples from loss in non-training phases
                        norm_mask = torch.isin(targets, normal_classes).to(device)
                    loss = loss_func(logits[norm_mask], norm_embeds[norm_mask], means, sigma2s,
                                     targets[norm_mask])
                # backward and optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optim.step()
                # novelty detection stats in non-training phase
                else:
                    cur_novel_scores = novelty_md(norm_embeds, means, sigma2s).detach().cpu()
                    novel_scores = torch.hstack([novel_scores, cur_novel_scores])
                    novel_labels = torch.hstack(
                        [novel_labels, torch.logical_not(norm_mask).int().detach().cpu()])
                # calculate statistics, masking novel classes
                _, preds = torch.max(logits[norm_mask], 1)
                epoch_loss = (loss.item() * data.size(0) +
                              cnt * epoch_loss) / (cnt + data.size(0))
                if len(preds) > 0:
                    epoch_acc = (torch.sum(preds == targets[norm_mask].data) +
                                 epoch_acc * cnt).double() / (cnt + data.size(0))
                    epoch_nll = (NDCCFixedLoss.nll_loss(
                        norm_embeds[norm_mask], means, sigma2s, targets[norm_mask]) +
                        epoch_nll * cnt) / (cnt + data.size(0))
                epoch_sigma2s = (torch.mean(sigma2s) * data.size(0) +
                                 epoch_sigma2s * cnt) / (cnt + data.size(0))
                cnt += data.size(0)
            # get phase label and update LR scheduler
            if phase == "train":
                phase_label = "Train"
                scheduler.step()
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
            # determine AUROC for non-training phases
            if phase != "train":
                auroc = roc_auc_score(novel_labels, novel_scores)
                writer.add_scalar(f"{phase_label}/NovDet AUROC", auroc, epoch)
                # record end of training stats, grouped as Metrics in Tensorboard
                if epoch == args.num_epochs - 1:
                    # note non-numeric values (NaN, None, ect.) will cause entry
                    # to not be displayed in Tensorboard HPARAMS tab
                    metric_dict.update({
                        f"Metrics/{phase_label}_loss": epoch_loss,
                        f"Metrics/{phase_label}_accuracy": epoch_acc,
                        f"Metrics/{phase_label}_nll": epoch_nll,
                        f"Metrics/{phase_label}_auroc": auroc,
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
    train_ndcc(args)
