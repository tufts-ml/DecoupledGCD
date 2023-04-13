from pathlib import Path

import torch


def cache_test_outputs(model, normal_classes, test_loader, out_dir):
    out_dir = Path(out_dir)
    model.eval()
    device = model.device
    # initialize caches
    out_logits = torch.empty((0, model.num_classes)).to(device)
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
        out_targets = torch.hstack((out_targets, targets))
        out_norm_mask = torch.hstack((out_norm_mask, norm_mask))
        if out_means is None or out_sigma2s is None:
            out_means = means.cpu()
            out_sigma2s = sigma2s.cpu()
    # write caches
    torch.save(out_logits.cpu(), out_dir / "logits.pt")
    torch.save(out_targets.cpu(), out_dir / "targets.pt")
    torch.save(out_norm_mask.cpu(), out_dir / "norm_mask.pt")
    torch.save(out_means.cpu(), out_dir / "means.pt")
    torch.save(out_sigma2s.cpu(), out_dir / "sigma2s.pt")
