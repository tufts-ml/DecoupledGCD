import argparse
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

from gcd_data.get_datasets import get_class_splits, get_datasets

from dpn.augment import sim_gcd_train, sim_gcd_test
from dpn.logger import AverageWriter
from dpn.loss import DPNLoss
from dpn.model import DPN
from dpn.scheduler import warm_cos_scheduler
from dpn.test.eval import cache_test_outputs, eval_from_cache


def get_args():
    parser = argparse.ArgumentParser()
    # dataset and split arguments
    parser.add_argument(
        "--dataset_name", type=str, default="cub",
        choices=["NovelCraft", "cifar10", "cifar100", "imagenet_100", "cub", "scars",
                 "aircraft", "herbarium_19"],
        help="options: NovelCraft, cifar10, cifar100, imagenet_100, cub, scars, aircraft, " +
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


def init_proto(model: DPN, train_loader, args, device):
    unlab_embeds = torch.empty((0, model.embed_len)).to(device)
    label_embeds = torch.empty((0, model.embed_len)).to(device)
    label_targets = torch.tensor([], dtype=int).to(device)
    unlab_idxs = torch.tensor([], dtype=int).to(device)
    # cache model outputs
    for batch in train_loader:
        data, targets, uq_idxs, label_mask = batch
        # move data to device
        data = data.to(device)
        targets = targets.long().to(device)
        uq_idxs = uq_idxs.to(device)
        label_mask = label_mask.to(device)
        with torch.set_grad_enabled(False):
            _, embeds = model(data)
            unlab_embeds = torch.vstack((unlab_embeds, embeds[~label_mask]))
            label_embeds = torch.vstack((label_embeds, embeds[label_mask]))
            label_targets = torch.hstack((label_targets, targets[label_mask]))
            unlab_idxs = torch.hstack((unlab_idxs, uq_idxs[~label_mask]))
    # construct labeled prototypes
    l_proto = torch.zeros((args.num_labeled_classes, model.embed_len))
    for i in range(args.num_labeled_classes):
        l_proto[i] = torch.mean(label_embeds[label_targets == i])
    # construct unlabeled prototypes
    km = KMeans(args.num_labeled_classes + args.num_unlabeled_classes, n_init=20).fit(
        unlab_embeds.cpu().numpy())
    init_u_proto = km.cluster_centers_
    # align unlabeled clusters to labeled clusters by shuffling indices to match
    cluster_dists = distance.cdist(l_proto, init_u_proto)
    row_ind, col_ind = linear_sum_assignment(cluster_dists)
    u_proto = torch.empty(init_u_proto.shape)
    for i in row_ind:
        u_proto[i] = torch.Tensor(init_u_proto[col_ind[i]])
    all_clusters = np.arange(args.num_labeled_classes + args.num_unlabeled_classes)
    unknown_clusters = torch.tensor(all_clusters[np.isin(all_clusters, col_ind)])
    for i in range(args.num_unlabeled_classes):
        u_proto[i + args.num_labeled_classes] = unknown_clusters[i]
    # find unlabeled known data based on K means labels (doesn't get updated)
    km.cluster_centers_ = u_proto.numpy()
    pseudo_labels = torch.tensor(km.predict(unlab_embeds.cpu().numpy()))
    uk_idxs = unlab_idxs[pseudo_labels < args.num_labeled_classes]
    # print split percents for unlabeled examples
    print(f"{100.0 * len(uk_idxs) / len(unlab_idxs)}% of unlabeled set pseudo-labeled as normal")
    return l_proto, u_proto, uk_idxs


def update_batch_stats(av_writer, phase, label_mask, label_types, loss_func: DPNLoss, targets,
                       preds, uk_mask):
    # calculate statistics masking unlabeled or novel data
    if label_mask.sum() > 0:
        av_writer.update(f"{phase}/Average {label_types[0]} Accuracy",
                         torch.mean((preds[label_mask] == targets[label_mask]).float()),
                         label_mask.sum())
        av_writer.update(f"{phase}/Average {label_types[0]} Cross-Entropy",
                         loss_func.last_k_loss,
                         label_mask.sum())
    # calculate statistics masking labeled or normal data
    if (~label_mask).sum() > 0:
        av_writer.update(f"{phase}/Average {label_types[1]} Total Loss",
                         loss_func.last_n_loss,
                         (~label_mask).sum())
        av_writer.update(f"{phase}/Average {label_types[1]} SPL Loss",
                         loss_func.last_spl_loss,
                         (~label_mask).sum())
        if uk_mask.sum() > 0:
            av_writer.update(f"{phase}/Average {label_types[1]} Transfer Loss",
                             loss_func.last_transfer_loss,
                             uk_mask.sum())


def update_cache_stats(av_writer, phase, label_types, label_embeds, unlab_embeds, l_proto, u_proto):
    # statistics for variance of embeddings and means
    # don't need third update parameter since only updated once before output
    av_writer.update(f"{phase}/Average {label_types[0]} Embedding Variance",
                     torch.mean(torch.var(label_embeds)))
    av_writer.update(f"{phase}/Average {label_types[1]} Embedding Variance",
                     torch.mean(torch.var(unlab_embeds)))
    av_writer.update(f"{phase}/{label_types[0]} Prototype Average Variance",
                     torch.mean(torch.var(l_proto)))
    av_writer.update(f"{phase}/{label_types[1]} Prototype Average Variance",
                     torch.mean(torch.var(u_proto)))


def train_gcd(args):
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # init dataloaders
    train_loader, valid_loader, test_loader, args = get_gcd_dataloaders(args)
    # normal classes after target transform according to GCDdatasets API
    normal_classes = torch.arange(args.num_labeled_classes).to(device)
    # init model
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    model = DPN(num_classes, args.num_labeled_classes, None, None).to(device)
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
    loss_func = DPNLoss()
    # init tensorboard, with random comment to stop overlapping runs
    av_writer = AverageWriter(args.label, comment=str(random.randint(0, 9999)))
    # metric dict for recording hparam metrics
    metric_dict = {}
    # initialze prototypes
    l_proto, u_proto, uk_idxs = init_proto(model, train_loader, args, device)
    model.l_proto = l_proto.to(device)
    model.u_proto = u_proto.to(device)
    del l_proto, u_proto
    # model training
    for epoch in range(args.num_epochs):
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
            # caches for updating labeled prototypes
            if phase == "Train":
                label_embeds = torch.empty((0, model.embed_len)).to(device)
                label_targets = torch.tensor([], dtype=int).to(device)
                # cache for logging only
                unlab_embeds = torch.empty((0, model.embed_len)).to(device)
            for batch in dataloader:
                # use label mask to separate labeled and unlabeled for train batches
                if phase == "Train":
                    data, targets, dataset_idxs, label_mask = (item.to(device) for item in batch)
                    label_types = ["Labeled", "Unlabeled"]
                # use label mask to separate normal and novel for non-train batches
                else:
                    data, targets, dataset_idxs = (item.to(device) for item in batch)
                    label_mask = torch.isin(targets.to(device), normal_classes)
                    label_types = ["Normal", "Novel"]
                # cast targets and count samples in batch
                targets = targets.long()
                num_samples = data.shape[0]
                # create unlabeled known mask, all False for validation set
                uk_mask = torch.isin(dataset_idxs, uk_idxs)
                # forward and loss
                optim.zero_grad()
                with torch.set_grad_enabled(phase == "Train"):
                    logits, embeds = model(data)
                    loss = loss_func(logits, embeds, targets, label_mask, uk_mask, model.l_proto,
                                     model.u_proto)
                # backward and optimize only if in training phase
                if phase == "Train":
                    loss.backward()
                    optim.step()
                    scheduler.step()
                # record non-class specific statistics
                av_writer.update(f"{phase}/Average Loss",
                                 loss.item(), num_samples)
                _, preds = torch.max(logits, 1)
                # cache data for updating labeled prototypes
                if phase == "Train":
                    label_embeds = torch.vstack((label_embeds, embeds[label_mask]))
                    label_targets = torch.hstack((label_targets, targets[label_mask]))
                    unlab_embeds = torch.vstack((unlab_embeds, embeds[~label_mask]))
                # output batch stats
                update_batch_stats(av_writer, phase, label_mask, label_types, loss_func, targets,
                                   preds, uk_mask)
            # update labeled prototypes
            if phase == "Train":
                model.update_l_proto(label_embeds, label_targets)
                # output cache stats
                update_cache_stats(av_writer, phase, label_types, label_embeds, unlab_embeds,
                                   model.l_proto, model.u_proto)
            # record end of training stats, grouped as Metrics in Tensorboard
            if phase != "Train" and epoch == args.num_epochs - 1:
                # note non-numeric values (NaN, None, ect.) will cause entry
                # to not be displayed in Tensorboard HPARAMS tab
                metric_dict.update({
                    f"Metrics/{phase}Loss": av_writer.get_avg(f"{phase}/Average Loss"),
                    f"Metrics/{phase}{label_types[0]}CE": av_writer.get_avg(
                        f"{phase}/Average {label_types[0]} Cross-Entropy"),
                    f"Metrics/{phase}{label_types[1]}Loss": av_writer.get_avg(
                        f"{phase}/Average {label_types[1]} Total Loss"),
                })
            # output statistics
            av_writer.write(epoch)
    # record hparams all at once and after all other writer calls
    # to avoid issues with Tensorboard changing output file
    av_writer.writer.add_hparams({
        "lr_e": args.lr_e,
        "lr_c": args.lr_c,
    }, metric_dict)
    out_dir = Path(av_writer.writer.get_logdir())
    torch.save(model.state_dict(), out_dir / f"{args.num_epochs}.pt")
    cache_test_outputs(model, normal_classes, test_loader, out_dir)
    eval_from_cache(out_dir)


if __name__ == "__main__":
    args = get_args()
    train_gcd(args)
