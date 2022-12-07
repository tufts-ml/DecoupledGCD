import torch
import torch.nn as nn
from torch.nn.functional import softplus


class DinoCCG(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.embed_len = self.dino.norm.normalized_shape[0]
        # linear classification head
        self.classifier = nn.Linear(self.embed_len, num_classes)
        # variance lower bound to prevent very small Gaussians
        self.var_min = 1e-3
        # solve inverse of variance calculation so variance is set to init_var
        init_var = torch.Tensor([1])
        init_sigma = torch.log(torch.exp(torch.sqrt(init_var - self.var_min)) - 1)
        # class-conditional Gaussian parameters
        self.sigma = nn.parameter.Parameter(init_sigma.type(torch.float32))
        self.deltas = nn.parameter.Parameter(torch.zeros((self.embed_len,), dtype=torch.float32))

    def gaussian_params(self):
        # class-conditional Gaussian parameters
        sigma2s = softplus(self.sigma + self.deltas)**2 + self.var_min
        means = self.classifier.weight * sigma2s
        return means, sigma2s

    def forward(self, x):
        # normalized DINO embeddings
        embeds = self.dino(x)
        norm_embeds = embeds.view(embeds.shape[0], -1)
        # classifier prediction
        logits = self.classifier(norm_embeds)
        means, sigma2s = self.gaussian_params()
        return logits, norm_embeds, means, sigma2s
