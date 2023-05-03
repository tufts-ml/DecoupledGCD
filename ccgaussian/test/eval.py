from pathlib import Path

import torch

from ccgaussian.test.bootstrap import bootstrap_gcd_acc
from ccgaussian.test.plot import plot_con_matrix, plot_gcd_ci
from ccgaussian.test.stats import cluster_confusion


def cache_test_outputs(model, normal_classes, test_loader, out_dir):
    out_dir = Path(out_dir)
    model.eval()
    # choose device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # initialize caches
    out_logits = torch.empty((0, model.num_classes)).to(device)
    out_embeds = torch.empty((0, model.embed_len)).to(device)
    out_targets = torch.tensor([], dtype=int).to(device)
    out_norm_mask = torch.tensor([], dtype=bool).to(device)
    out_means = None
    out_sigma2s = None
    for data, targets, uq_idxs in test_loader:
        # create normal mask
        norm_mask = torch.isin(targets.to(device), normal_classes).to(device)
        # move data to device
        data = data.to(device)
        targets = targets.long().to(device)
        # forward pass
        with torch.set_grad_enabled(False):
            logits, embeds, means, sigma2s = model(data)
        # cache data
        out_logits = torch.vstack((out_logits, logits))
        out_embeds = torch.vstack((out_embeds, embeds))
        out_targets = torch.hstack((out_targets, targets))
        out_norm_mask = torch.hstack((out_norm_mask, norm_mask))
        if out_means is None or out_sigma2s is None:
            out_means = means.cpu()
            out_sigma2s = sigma2s.cpu()
    # write caches
    torch.save(out_logits.cpu(), out_dir / "logits.pt")
    torch.save(out_embeds.cpu(), out_dir / "embeds.pt")
    torch.save(out_targets.cpu(), out_dir / "targets.pt")
    torch.save(out_norm_mask.cpu(), out_dir / "norm_mask.pt")
    torch.save(out_means.cpu(), out_dir / "means.pt")
    torch.save(out_sigma2s.cpu(), out_dir / "sigma2s.pt")


def eval_from_cache(out_dir):
    out_dir = Path(out_dir)
    y_pred = torch.max(torch.load(out_dir / "logits.pt"), dim=1)[1].numpy()
    y_true = torch.load(out_dir / "targets.pt").numpy()
    norm_mask = torch.load(out_dir / "norm_mask.pt").numpy()
    # get accuracies
    plot_gcd_ci(
        *bootstrap_gcd_acc(y_pred, y_true, norm_mask)
    ).savefig(out_dir / "acc_ci.png")
    plot_con_matrix(cluster_confusion(y_pred, y_true)).savefig(out_dir / "conf_mat.png")
