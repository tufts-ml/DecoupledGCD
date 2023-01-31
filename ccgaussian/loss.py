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
    def mahalanobis_d(norm_embeds, means, sigma2s, targets):
        return ((norm_embeds - means[targets])**2 / sigma2s).sum() / \
            (2 * norm_embeds.shape[0])

    @staticmethod
    def nll_loss(norm_embeds, means, sigma2s, targets):
        # negative log-likelihood loss
        return torch.log(sigma2s).sum() / 2 + \
            NDCCLoss.mahalanobis_d(norm_embeds, means, sigma2s, targets)

    def forward(self, logits, norm_embeds, means, sigma2s, targets):
        if norm_embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        return self.ce_loss(logits, targets) + \
            self.w_nll * self.nll_loss(norm_embeds, means, sigma2s, targets)


class UnsupMDLoss(torch.nn.Module):
    def __init__(self, w_ccg) -> None:
        super().__init__()
        self.w_ccg = w_ccg

    @staticmethod
    def unsup_md_loss(norm_embeds_1, norm_embeds_2, sigma2s):
        return ((norm_embeds_1 - norm_embeds_2)**2 / sigma2s.detach()).sum() / \
            (2 * norm_embeds_1.shape[0])

    def forward(self, norm_embeds_1, norm_embeds_2, sigma2s):
        return self.w_ccg * self.unsup_md_loss(norm_embeds_1, norm_embeds_2, sigma2s)


def novelty_md(norm_embeds, means, sigma2s):
    # goes from B x K x D to B
    all_md = torch.sum((torch.unsqueeze(norm_embeds, dim=1) - means)**2 / sigma2s, dim=2) / \
        (2 * norm_embeds.shape[0])
    return torch.min(all_md, dim=1)[0]
