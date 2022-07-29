from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay
import torch

from polycraft_nov_data.dataloader import novelcraft_dataloader

from ccgaussian.dino_trans import DINOTestTrans
from ccgaussian.load import load_dino_ccg


def sq_mahalanobis_dist(embeds, means, sigma2s):
    """Squared Mahalanobis distance

    Args:
        embeds (torch.Tensor): Embeddings (N, D)
        means (torch.Tensor): Means (K, D)
        sigma2s (torch.Tensor): Diagonal of covariance matrix (D,)

    Returns:
        torch.Tensor: Distances between embedding n and mean k, (N, K)
    """
    # N x K x D
    difs = torch.unsqueeze(embeds, 1) - torch.unsqueeze(means, 0)
    # D x D
    inv_cov = torch.diag_embed(1 / sigma2s)
    # N x K x 1 x D @ D x D @ N x K x D x 1 = N x K x 1 x 1
    dists = torch.unsqueeze(difs, 2) @ inv_cov @ torch.unsqueeze(difs, 3)
    # N x K
    return torch.squeeze(dists)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = load_dino_ccg().to(device)
    valid_dataloader = novelcraft_dataloader("valid_norm", DINOTestTrans(), batch_size=128)
    classifier_pred = torch.tensor([])
    ccg_pred = torch.tensor([])
    true_targets = torch.tensor([])
    for data, targets in valid_dataloader:
        with torch.no_grad():
            logits, norm_embeds, means, sigma2s = model(data.to(device))
            logits = logits.to("cpu")
            dists = sq_mahalanobis_dist(norm_embeds, means, sigma2s).to("cpu")
            classifier_pred = torch.hstack((classifier_pred, torch.argmax(logits, 1)))
            ccg_pred = torch.hstack((ccg_pred, torch.argmin(dists, 1)))
            true_targets = torch.hstack((true_targets, targets))
    # print stats
    print(f"Classifier accuracy: {torch.sum(classifier_pred == true_targets) / len(true_targets)}")
    print(f"CCG accuracy: {torch.sum(ccg_pred == true_targets) / len(true_targets)}")
    print(f"Model agreement: {torch.sum(classifier_pred == ccg_pred) / len(true_targets)}")
    # save confusion matrices
    output_dir = Path("runs/DinoCCG_0_0_1")
    ConfusionMatrixDisplay.from_predictions(true_targets, classifier_pred) \
        .plot().savefig(output_dir / "class_con_mat.png")
    ConfusionMatrixDisplay.from_predictions(true_targets, ccg_pred) \
        .plot().savefig(output_dir / "ccg_con_mat.png")
