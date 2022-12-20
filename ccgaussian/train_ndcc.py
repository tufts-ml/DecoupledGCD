import argparse
from pathlib import Path

from sklearn.metrics import roc_auc_score
import torch
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data.dataloader import novelcraft_dataloader

from ccgaussian.dino_trans import DINOTestTrans
from ccgaussian.loss import NDCCLoss, novelty_md
from ccgaussian.model import DinoCCG


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NovelCraft", choices=["NovelCraft"])
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
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_milestones", default=[25, 40, 45])
    # loss hyperparameters
    parser.add_argument("--w_nll", type=float, default=1e-2,
                        help="Negative log-likelihood weight for embedding network")
    args = parser.parse_args()
    # add dataset related args
    if args.dataset == "NovelCraft":
        args.num_classes = 5
    # prepend runs folder to label if given
    if args.label is not None:
        args.label = "runs/" + args.label
    return args


def train_ndcc(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # init dataloaders
    if args.dataset == "NovelCraft":
        train_loader = novelcraft_dataloader("train", DINOTestTrans(), args.batch_size,
                                             balance_classes=True)
        valid_norm_loader = novelcraft_dataloader("valid_norm", DINOTestTrans(), args.batch_size,
                                                  balance_classes=False)
        valid_nov_loader = novelcraft_dataloader("valid_novel", DINOTestTrans(), args.batch_size,
                                                 balance_classes=False)
    # init model
    model = DinoCCG(args.num_classes, args.init_var).to(device)
    # init optimizer
    optim = torch.optim.SGD([
        {
            "params": model.dino.parameters(),
            "lr": args.lr_e,
            "weignt_decay": 5e-4,
            "momentum": args.momentum,
        },
        {
            "params": model.classifier.parameters(),
            "lr": args.lr_c,
            "momentum": args.momentum,
        },
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=args.lr_milestones, gamma=0.1)
    # init loss
    loss_func = NDCCLoss(args.w_nll)
    # init tensorboard
    writer = SummaryWriter(args.label)
    # model training
    for epoch in range(args.num_epochs):
        epoch_novel_scores = torch.Tensor([])
        epoch_novel_labels = torch.Tensor([])
        # Each epoch has a training and validation phase
        for phase in ["train", "val_norm", "val_nov"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            elif phase == "val_norm":
                model.eval()
                dataloader = valid_norm_loader
            else:
                model.eval()
                dataloader = valid_nov_loader
            # vars for tensorboard stats
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            epoch_nll = 0.
            sigma2s_mean = 0.
            for data, targets in dataloader:
                # forward and loss
                data = (data.to(device))
                targets = (targets.long().to(device))
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits, norm_embeds, means, sigma2s = model(data)
                    if phase != "val_nov":
                        loss = loss_func(logits, norm_embeds, means, sigma2s, targets)
                # backward and optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optim.step()
                # collect novelty detection stats only if validation phase
                else:
                    novel_scores = novelty_md(norm_embeds, means, sigma2s).detach().cpu()
                    epoch_novel_scores = torch.hstack([epoch_novel_scores, novel_scores])
                    epoch_novel_labels = torch.hstack([epoch_novel_labels, torch.Tensor(
                        [1 if phase == "val_nov" else 0] * len(novel_scores))])
                # calculate statistics
                if phase != "val_nov":
                    _, preds = torch.max(logits, 1)
                    epoch_loss = (loss.item() * data.size(0) +
                                  cnt * epoch_loss) / (cnt + data.size(0))
                    epoch_acc = (torch.sum(preds == targets.data) +
                                 epoch_acc * cnt).double() / (cnt + data.size(0))
                    epoch_nll = (NDCCLoss.nll_loss(norm_embeds, means, sigma2s, targets) +
                                 epoch_nll * cnt) / (cnt + data.size(0))
                    sigma2s_mean = (torch.mean(sigma2s) * data.size(0) +
                                    sigma2s_mean * cnt) / (cnt + data.size(0))
                cnt += data.size(0)
            if phase == "train":
                phase_label = "Train"
                scheduler.step()
            else:
                phase_label = "Valid"
            # output statistics
            if phase != "val_nov":
                writer.add_scalar(f"{phase_label}/Average Loss", epoch_loss, epoch)
                writer.add_scalar(f"{phase_label}/Average Accuracy", epoch_acc, epoch)
                writer.add_scalar(f"{phase_label}/Average NLL", epoch_nll, epoch)
        # determine validation AUROC after all phases
        epoch_auroc = roc_auc_score(epoch_novel_labels, epoch_novel_scores)
        writer.add_scalar(f"{phase_label}/NovDet AUROC", epoch_auroc, epoch)
    writer.add_hparams({
        "lr_e": args.lr_e,
        "lr_c": args.lr_c,
        "init_var": args.init_var,
        "w_nll": args.w_nll,
    }, {
        "hparam/val_loss": epoch_loss,
        "hparam/val_accuracy": epoch_acc,
        "hparam/val_nll": epoch_nll,
        "hparam/val_auroc": epoch_auroc,
    })
    torch.save(model.state_dict(), Path(writer.get_logdir()) / f"{args.num_epochs}.pt")


if __name__ == "__main__":
    args = get_args()
    train_ndcc(args)
