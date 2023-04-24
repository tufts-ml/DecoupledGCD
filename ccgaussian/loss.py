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
            torch.mean(NDCCLoss.sq_mahalanobis_d(embeds, means, sigma2s, targets))


class GMMFixedLoss(NDCCLoss):
    def __init__(self, w_nll, w_unlab, pseudo_thresh) -> None:
        super().__init__(w_nll)
        self.w_unlab = w_unlab
        self.pseudo_thresh = pseudo_thresh

    def embed_md_loss(self, embeds, sigma2s, gmm_means, soft_targets):
        return NDCCLoss.sq_mahalanobis_d(
            embeds, gmm_means, sigma2s, soft_targets.argmax(dim=1)).mean()

    def means_md_loss(self, means, sigma2s, gmm_means, soft_targets):
        p_targets = soft_targets.argmax(dim=1)
        return NDCCLoss.sq_mahalanobis_d(means[p_targets], gmm_means, sigma2s, p_targets).mean()

    def md_loss(self, embeds, means, sigma2s, gmm_means, soft_targets):
        embed_md = self.embed_md_loss(embeds, sigma2s, gmm_means, soft_targets)
        means_md = self.means_md_loss(means, sigma2s, gmm_means, soft_targets)
        return embed_md + means_md

    # NDCCLoss for soft labels and fixed variance
    def forward(self, logits, embeds, means, sigma2s, gmm_means, soft_targets, label_mask):
        if embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        # validate soft_targets
        assert torch.allclose(torch.sum(soft_targets, axis=1),
                              torch.ones(soft_targets.shape[0]).to(soft_targets.device))
        # normal loss, checking for empty inputs
        if label_mask.sum() > 0:
            norm_l = self.ce_loss(logits[label_mask], soft_targets[label_mask]) + \
                self.w_nll * self.md_loss(embeds[label_mask], means, sigma2s, gmm_means,
                                          soft_targets[label_mask])
        else:
            norm_l = 0
        # create mask for unlabeled data accounting for pseudo-label thresholding
        unlabel_mask = torch.logical_and(
            ~label_mask, soft_targets.max(axis=1)[0] >= self.pseudo_thresh)
        # novel loss, checking for empty inputs
        if unlabel_mask.sum() > 0:
            novel_l = self.ce_loss(logits[unlabel_mask], soft_targets[unlabel_mask]) + \
                self.w_nll * self.md_loss(embeds[unlabel_mask], means, sigma2s, gmm_means,
                                          soft_targets[unlabel_mask])
        else:
            novel_l = 0
        return (1 - self.w_unlab) * norm_l + self.w_unlab * novel_l


def all_sq_md(embeds, means, sigma2s):
    # Mahalanobis distance for embeddings to each class
    # goes from B x K x D to B x K
    return torch.sum((torch.unsqueeze(embeds, dim=1) - means)**2 / sigma2s, dim=2)


def novelty_sq_md(embeds, means, sigma2s):
    # Mahalanobis distance for embeddings to closest class
    # goes from B x K to B
    return torch.min(all_sq_md(embeds, means, sigma2s), dim=1)[0]
