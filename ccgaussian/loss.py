import torch
import torch.nn
import torch.nn.functional as f


class NDCCLoss(torch.nn.Module):
    def __init__(self, w_nll) -> None:
        super().__init__()
        self.w_nll = w_nll

    @staticmethod
    def ce_loss(logits, targets):
        return f.cross_entropy(logits, targets)

    @staticmethod
    def sq_mahalanobis_d(embeds, means, sigma2s, targets):
        # goes from B x D to B
        return ((embeds - means[targets])**2 / sigma2s).sum(dim=1)

    @staticmethod
    def nll_loss(embeds, means, sigma2s, targets):
        # negative log-likelihood loss
        return torch.log(sigma2s).sum() / 2 + \
            torch.mean(NDCCLoss.sq_mahalanobis_d(embeds, means, sigma2s, targets)) / 2

    def forward(self, logits, embeds, means, sigma2s, targets):
        if embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        return self.ce_loss(logits, targets) + \
            self.w_nll * self.nll_loss(embeds, means, sigma2s, targets)


class NDCCFixedLoss(NDCCLoss):
    # NDCCLoss for fixed variance
    def forward(self, logits, embeds, means, sigma2s, targets):
        if embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        return self.ce_loss(logits, targets) + self.w_nll * \
            torch.mean(NDCCLoss.sq_mahalanobis_d(embeds, means, sigma2s, targets)) / 2


class NDCCFixedSoftLoss(NDCCLoss):
    # NDCCLoss for soft labels and fixed variance
    def forward(self, logits, embeds, means, sigma2s, soft_targets):
        if embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        # validate soft_targets
        assert torch.allclose(torch.sum(soft_targets, axis=1), torch.ones(soft_targets.shape[0]))
        # take sum over clusters then mean over batch dimension
        md_reg = torch.sum(all_sq_md(embeds, means, sigma2s) * soft_targets, dim=1).mean() / 2
        return self.ce_loss(logits, soft_targets) + self.w_nll * md_reg


def all_sq_md(embeds, means, sigma2s):
    # Mahalanobis distance for embeddings to each class
    # goes from B x K x D to B x K
    return torch.sum((torch.unsqueeze(embeds, dim=1) - means)**2 / sigma2s, dim=2)


def novelty_sq_md(embeds, means, sigma2s):
    # Mahalanobis distance for embeddings to closest class
    # goes from B x K to B
    return torch.min(all_sq_md(embeds, means, sigma2s), dim=1)[0]
