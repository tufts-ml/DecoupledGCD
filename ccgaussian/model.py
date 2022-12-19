import torch
import torch.nn as nn


class DinoCCG(nn.Module):
    def __init__(self, num_classes, variance=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.embed_len = self.dino.norm.normalized_shape[0]
        # linear classification head
        self.classifier = nn.Linear(self.embed_len, num_classes)
        # class-conditional Gaussian parameters
        self.sigma2s = nn.parameter.Parameter(
            torch.Tensor([variance] * self.embed_len), requires_grad=False)

    def gaussian_params(self):
        # class-conditional Gaussian parameters
        means = self.classifier.weight * self.sigma2s
        return means, self.sigma2s

    def forward(self, x):
        # DINO embeddings
        raw_embeds = self.dino(x)
        embeds = raw_embeds.view(raw_embeds.shape[0], -1)
        # classifier prediction
        logits = self.classifier(embeds)
        means, sigma2s = self.gaussian_params()
        return logits, embeds, means, sigma2s
