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
    def sq_mahalanobis_d(norm_embeds, means, sigma2s, targets):
        # goes from B x D to B
        return ((norm_embeds - means[targets])**2 / sigma2s).sum(dim=1)

    @staticmethod
    def nll_loss(norm_embeds, means, sigma2s, targets):
        # negative log-likelihood loss
        return torch.log(sigma2s).sum() / 2 + \
            torch.mean(NDCCLoss.sq_mahalanobis_d(norm_embeds, means, sigma2s, targets)) / 2

    def forward(self, logits, norm_embeds, means, sigma2s, targets):
        if norm_embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        return self.ce_loss(logits, targets) + \
            self.w_nll * self.nll_loss(norm_embeds, means, sigma2s, targets)


class NDCCFixedLoss(NDCCLoss):
    # NDCCLoss for fixed variance
    def forward(self, logits, norm_embeds, means, sigma2s, targets):
        if norm_embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        return self.ce_loss(logits, targets) + self.w_nll * \
            torch.mean(NDCCLoss.sq_mahalanobis_d(norm_embeds, means, sigma2s, targets)) / 2


def novelty_sq_md(norm_embeds, means, sigma2s):
    # goes from B x K x D to B x K to B
    all_md = torch.sum((torch.unsqueeze(norm_embeds, dim=1) - means)**2 / sigma2s, dim=2)
    return torch.min(all_md, dim=1)[0]
