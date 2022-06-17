import torch
import torch.nn as nn
from torch.nn.functional import normalize


class DinoCCG(nn.Module):
    def __init__(self, num_classes, embed_mag=16.) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_mag = embed_mag
        self.embed_len = 4096
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        # linear classification head
        self.classifier = nn.Linear(self.embed_len, num_classes)
        # class-conditional Gaussian parameters
        self.sigma = torch.ones((1,), dtype=torch.float32, requires_grad=True)
        self.delta = torch.zeros((self.embed_len,), dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        # normalized DINO embeddings
        embeds = self.dino(x)
        norm_embeds = self.embed_mag * normalize(embeds.view(embeds.shape[0], -1), dim=1)
        # classifier prediction
        logits = self.classifier(norm_embeds)
        # class-conditional Gaussian parameters
        sigma2s = (self.sigma + self.delta) ** 2
        means = self.classifier.weight * sigma2s
        return logits, norm_embeds, means, sigma2s
