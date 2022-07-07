import torch
import torch.nn as nn
from torch.nn.functional import normalize


class DinoCCG(nn.Module):
    def __init__(self, num_classes, embed_mag=16.) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_mag = embed_mag
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.embed_len = self.dino.norm.normalized_shape[0]
        # linear classification head
        self.classifier = nn.Linear(self.embed_len, num_classes)
        # class-conditional Gaussian parameters
        self.sigma = torch.ones((1,), dtype=torch.float32, requires_grad=True)
        self.deltas = torch.zeros((self.embed_len,), dtype=torch.float32, requires_grad=True)
        # variance lower bound to prevent very large losses
        self.var_min = .1

    def gaussian_params(self):
        # class-conditional Gaussian parameters
        sigma2s = (self.sigma + self.deltas)**2 + self.var_min
        means = self.classifier.weight * sigma2s
        return means, sigma2s

    def forward(self, x):
        # normalized DINO embeddings
        embeds = self.dino(x)
        norm_embeds = self.embed_mag * normalize(embeds.view(embeds.shape[0], -1), dim=1)
        # classifier prediction
        logits = self.classifier(norm_embeds)
        means, sigma2s = self.gaussian_params()
        return logits, norm_embeds, means, sigma2s
