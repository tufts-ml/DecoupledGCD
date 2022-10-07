import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data.dataloader import novelcraft_dataloader

from ccgaussian.dino_trans import DINOTestTrans
from ccgaussian.loss import NDCCLoss
from ccgaussian.model import DinoCCG


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NovelCraft", choices=["NovelCraft"])
    # model label for logging
    parser.add_argument("--label", type=str, default=None)
    # model hyperparameters
    parser.add_argument("--e_mag", type=float, default=16, help="Embedding magnitued")
    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_e", type=float, default=1e-3,
                        help="Learning rate for embedding v(x)")
    parser.add_argument("--lr_c", type=float, default=1e-1,
                        help="Learning rate for linear classifier {w_y, b_y}")
    parser.add_argument("--lr_s", type=float, default=1e-1,
                        help="Learning rate for sigma")
    parser.add_argument("--lr_d", type=float, default=1e-3,
                        help="Learning rate for delta_j")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_milestones", default=[15, 25, 29])
    # loss hyperparameters
    parser.add_argument("--w_nll", type=float, default=2e-1,
                        help="Negative log-likelihood weight, gamma in Eq. (22)")
    args = parser.parse_args()
    # add dataset related args
    if args.dataset == "NovelCraft":
        args.num_classes = 5
    return args


def train_ndcc(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # init dataloaders
    if args.dataset == "NovelCraft":
        train_loader = novelcraft_dataloader("train", DINOTestTrans(), args.batch_size,
                                             balance_classes=True)
        valid_loader = novelcraft_dataloader("valid_norm", DINOTestTrans(), args.batch_size,
                                             balance_classes=False)
    # init model
    model = DinoCCG(args.num_classes, args.e_mag).to(device)
    # init optimizer
    optim = torch.optim.SGD([
        {
            "params": model.dino.parameters(),
            "lr": args.lr_e,
            "weignt_decay": 5e-4,
        },
        {
            "params": model.classifier.parameters(),
            "lr": args.lr_c,
        },
        {
            "params": [model.sigma],
            "lr": args.lr_s,
        },
        {
            "params": [model.deltas],
            "lr": args.lr_d,
        },
    ], momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=args.lr_milestones, gamma=0.1)
    # init loss
    loss_func = NDCCLoss(args.w_nll)
    # init tensorboard
    writer = SummaryWriter(args.label)
    # model training
    for epoch in range(args.num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader
            # vars for tensorboard stats
            cnt = 0
            epoch_loss = 0.
            epoch_acc = 0.
            epoch_nll = 0.
            sigma2s_norm = 0.
            for data, targets in dataloader:
                # forward and loss
                data = (data.to(device))
                targets = (targets.long().to(device))
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits, norm_embeds, means, sigma2s = model(data)
                    loss = loss_func(logits, norm_embeds, means, sigma2s, targets)
                # backward and optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optim.step()
                # statistics
                _, preds = torch.max(logits, 1)
                epoch_loss = (loss.item() * data.size(0) +
                              cnt * epoch_loss) / (cnt + data.size(0))
                epoch_acc = (torch.sum(preds == targets.data) +
                             epoch_acc * cnt).double() / (cnt + data.size(0))
                epoch_nll = (NDCCLoss.nll_loss(norm_embeds, means, sigma2s, targets) +
                             epoch_nll * cnt) / (cnt + data.size(0))
                sigma2s_norm = (torch.mean(sigma2s) * data.size(0) +
                                sigma2s_norm * cnt) / (cnt + data.size(0))
                cnt += data.size(0)
            if phase == "train":
                scheduler.step()
                writer.add_scalar("Average Train Loss", epoch_loss, epoch)
                writer.add_scalar("Average Train Accuracy", epoch_acc, epoch)
                writer.add_scalar("Average Train NLL", epoch_nll, epoch)
                writer.add_scalar("Average Variance Mean", sigma2s_norm, epoch)
            else:
                writer.add_scalar("Average Valid Loss", epoch_loss, epoch)
                writer.add_scalar("Average Valid Accuracy", epoch_acc, epoch)
                writer.add_scalar("Average Valid NLL", epoch_nll, epoch)
    torch.save(model.state_dict(), Path(writer.get_logdir()) / f"{args.num_epochs}.pt")


if __name__ == "__main__":
    args = get_args()
    train_ndcc(args)
