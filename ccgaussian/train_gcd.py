import argparse
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from polycraft_nov_data.dataloader import novelcraft_dataloader

from ccgaussian.dino_trans import DINOTestTrans, DINOConsistentTrans
from ccgaussian.loss import NDCCLoss, UnsupMDLoss
from ccgaussian.model import DinoCCG


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NovelCraft", choices=["NovelCraft"])
    # device parameters
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device", type=int, help="CUDA device index or unused for CPU")
    device_group.add_argument("--parallel", action="store_true", help="Enable multi-GPU training")
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
    parser.add_argument("--w_ccg", type=float, default=2e-1,
                        help="CCG loss weight, lambda in Eq. (23)")
    parser.add_argument("--w_nll", type=float, default=1 / 4096,
                        help="Negative log-likelihood weight, gamma in Eq. (22)")
    args = parser.parse_args()
    # add dataset related args
    if args.dataset == "NovelCraft":
        args.num_classes = 5
    return args


def train_gcd(args):
    # choose device
    device = torch.device(f"cuda:{args.device}") if args.device is not None else torch.device("cpu")
    # init dataloaders
    if args.dataset == "NovelCraft":
        sup_train_loader = novelcraft_dataloader("train", DINOTestTrans(), args.batch_size,
                                                 balance_classes=True)
        unsup_train_loader = novelcraft_dataloader("valid", DINOConsistentTrans(), args.batch_size)
        sup_valid_loader = novelcraft_dataloader("test_norm", DINOTestTrans(), args.batch_size,
                                                 balance_classes=False)
        # unsup iterator to have unaligned epochs
        unsup_iter = iter(unsup_train_loader)
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
    # convert model for parallel processing
    if args.parallel:
        model = DistributedDataParallel(model, [args.device])
    # init loss
    sup_loss_func = NDCCLoss(args.w_ccg, args.w_nll)
    unsup_loss_func = UnsupMDLoss(args.w_ccg)
    # init tensorboard
    writer = SummaryWriter()
    # model training
    for epoch in range(args.num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                sup_loader = sup_train_loader
            else:
                model.eval()
                sup_loader = sup_valid_loader
            # vars for tensorboard stats
            sup_count = 0
            unsup_count = 0
            epoch_sup_loss = 0.
            epoch_unsup_loss = 0.
            epoch_acc = 0.
            for data, targets in sup_loader:
                # supervised forward and loss
                data = (data.to(device))
                targets = (targets.long().to(device))
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits, norm_embeds, means, sigma2s = model(data)
                    sup_loss = sup_loss_func(logits, norm_embeds, means, sigma2s, targets)
                # supervised stats
                _, preds = torch.max(logits, 1)
                epoch_sup_loss = (sup_loss.item() * data.size(0) +
                                  sup_count * epoch_sup_loss) / (sup_count + data.size(0))
                epoch_acc = (torch.sum(preds == targets.data) +
                             epoch_acc * sup_count).double() / (sup_count + data.size(0))
                sup_count += data.size(0)
                # unsupervised forward, loss, and backprop
                unsup_loss = 0
                if phase == "train":
                    # get unlabeled batch
                    try:
                        (u_data, u_t_data), _ = next(unsup_iter)
                    except StopIteration:
                        unsup_iter = iter(unsup_train_loader)
                        (u_data, u_t_data), _ = next(unsup_iter)
                    # unsupervised forward and loss
                    u_data, u_t_data = u_data.to(device), u_t_data.to(device)
                    _, u_norm_embeds, _, sigma2s = model(u_data)
                    _, u_t_norm_embeds, _, _ = model(u_t_data)
                    unsup_loss = unsup_loss_func(u_norm_embeds, u_t_norm_embeds, sigma2s)
                    # backward and optimize
                    loss = sup_loss + unsup_loss
                    loss.backward()
                    optim.step()
                    # unsupervised stats
                    epoch_unsup_loss = (
                        unsup_loss.item() * u_data.size(0) + unsup_count * epoch_unsup_loss
                        ) / (unsup_count + u_data.size(0))
                    unsup_count += u_data.size(0)
            if phase == "train":
                scheduler.step()
                writer.add_scalar("Average Train Supervised Loss", epoch_sup_loss, epoch)
                writer.add_scalar("Average Train Unsupervised Loss", epoch_unsup_loss, epoch)
                writer.add_scalar("Average Train Accuracy", epoch_acc, epoch)
            else:
                writer.add_scalar("Average Valid Supervised Loss", epoch_sup_loss, epoch)
                writer.add_scalar("Average Valid Accuracy", epoch_acc, epoch)
    torch.save(model.state_dict(), Path(writer.get_logdir()) / f"{args.num_epochs}.pt")


def launch_parallel(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    args.device = rank
    args.batch_size = args.batch_size // world_size
    train_gcd(args)


if __name__ == "__main__":
    args = get_args()
    if not args.parallel:
        train_gcd(args)
    else:
        world_size = 2
        torch.multiprocessing.spawn(launch_parallel, (world_size, args), world_size)
