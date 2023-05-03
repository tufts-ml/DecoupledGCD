import torch
import torch.nn as nn


class DinoCCG(nn.Module):
    def __init__(self, num_classes, init_var=1, end_var=.3, var_warmup=25) -> None:
        super().__init__()
        self.num_classes = num_classes
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.embed_len = self.dino.norm.normalized_shape[0]
        # linear classification head
        self.classifier = nn.Linear(self.embed_len, num_classes)
        # class-conditional Gaussian parameters
        self.sigma2s = nn.parameter.Parameter(
            torch.Tensor([init_var] * self.embed_len), requires_grad=False)
        # variance annealing parameters
        self.init_var = init_var
        self.end_var = end_var
        self.var_warmup = var_warmup

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

    def anneal_var(self, epoch_num):
        # determine factors for interpolation between init and end
        epoch_factor = min(epoch_num / self.var_warmup, 1)
        anneal_factor = float((1 + torch.cos(torch.scalar_tensor(epoch_factor * torch.pi))) / 2)
        # update variance, applying cos annealing in 1/x space used by loss then mapping to x space
        recip_init = 1 / self.init_var
        recip_end = 1 / self.end_var
        new_var = 1 / (recip_end + (recip_init - recip_end) * anneal_factor)
        self.sigma2s[:] = new_var
