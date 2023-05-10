import torch
import torch.nn
import torch.nn.functional as f


class DPNLoss(torch.nn.Module):
    def __init__(self, temp=0.07, transfer_weight=10) -> None:
        self.temp = temp
        self.transfer_weight = transfer_weight
        super().__init__()

    @staticmethod
    def ce_loss(logits, targets):
        return f.cross_entropy(logits, targets)

    @staticmethod
    def cos_sim(embeds, proto):
        # add empty dimensions for easy broadcasting
        embeds = embeds.unsqueeze(1)  # B x 1 x D
        proto = proto.unsqueeze(0)  # 1 x K x D
        # similarity to each prototype B x K
        sim = f.cosine_similarity(embeds, proto, dim=2)
        return sim

    def norm_cos_sim(self, embeds, proto):
        # similarity to each prototype B x K
        sim = self.cos_sim(embeds, proto)
        # temp softmax across clusters B x K
        sim_norm = torch.softmax(sim / self.temp, dim=1)
        return sim_norm

    @staticmethod
    def l2_dist(embeds, proto):
        # add empty dimensions for easy broadcasting
        embeds = embeds.unsqueeze(1)  # B x 1 x D
        proto = proto.unsqueeze(0)  # 1 x K x D
        # distance to each prototype B x K
        dist = torch.norm(embeds - proto, p=2, dim=2)
        return dist

    def spl_loss(self, u_embeds: torch.Tensor, u_proto: torch.Tensor):
        """Semantic-aware Prototypical Learning loss

        Args:
            u_embeds (torch.Tensor): Unlabeled embeddings B x D
            u_proto (torch.Tensor): Unlabeled prototypes K x D

        Returns:
            torch.Tensor: Scalar loss, averaged over B
        """
        # temp softmax across clusters B x K
        sim_norm = self.norm_cos_sim(u_embeds, u_proto)
        # distance to each prototype B x K
        dist = self.l2_dist(u_embeds, u_proto)
        # sum over K and take mean over B to get scalar
        return torch.sum(dist * sim_norm, dim=1).mean()

    def transfer_loss(self, uk_embeds: torch.Tensor, l_proto: torch.Tensor):
        """Semantic-aware Prototypical Learning loss

        Args:
            uk_embeds (torch.Tensor): Unlabeled known embeddings B x D
            l_proto (torch.Tensor): Labeled prototypes K x D

        Returns:
            torch.Tensor: Scalar loss, averaged over B
        """
        # temp softmax across clusters B x K
        sim_norm = self.norm_cos_sim(uk_embeds, l_proto)
        # distance to each prototype B x K
        dist = 1 - self.cos_sim(uk_embeds, l_proto)
        # sum over K and take mean over B to get scalar
        return torch.sum(dist * sim_norm, dim=1).mean()

    def forward(self, logits, embeds, targets, label_mask, uk_mask, l_proto, u_proto):
        n_loss = 0
        k_loss = 0
        # cross entropy loss on labeled data only
        if label_mask.sum() > 0:
            k_loss += self.ce_loss(logits[label_mask], targets[label_mask])
        # SPL and transfer losses on unlabeled data only
        if (~label_mask).sum() > 0:
            n_loss += self.spl_loss(embeds[~label_mask], u_proto)
            self.last_spl_loss = n_loss
            if uk_mask.sum() > 0:
                n_loss += self.transfer_weight * self.transfer_loss(embeds[uk_mask], l_proto)
                self.last_transfer_loss = (n_loss - self.last_spl_loss)
        # store loss values for logging
        self.last_n_loss = n_loss
        self.last_k_loss = k_loss
        return n_loss + k_loss
