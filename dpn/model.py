import torch
import torch.nn as nn


class DPN(nn.Module):
    def __init__(self, num_classes, num_labeled_classes, l_proto, u_proto, l_moment=.9) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_labeled_classes = num_labeled_classes
        self.l_moment = l_moment
        # pretrained DINO backbone
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.embed_len = self.dino.norm.normalized_shape[0]
        # linear classification head for labeled classes only
        self.classifier = nn.Linear(self.embed_len, num_labeled_classes)
        # initialize prototypes after setting up other parameters
        self.set_protos(l_proto, u_proto)

    def forward(self, x):
        # DINO embeddings
        raw_embeds = self.dino(x)
        embeds = raw_embeds.view(raw_embeds.shape[0], -1)
        # classifier prediction
        logits = self.classifier(embeds)
        return logits, embeds

    def set_protos(self, l_proto, u_proto):
        device = next(self.parameters()).device
        self.l_proto = nn.Parameter(l_proto.detach().to(device), requires_grad=False)
        self.u_proto = nn.Parameter(u_proto.detach().to(device), requires_grad=False)

    def update_l_proto(self, label_embeds, label_targets):
        new_l_proto = torch.zeros_like(self.l_proto)
        for i in torch.unique(label_targets):
            new_l_proto[i] = torch.mean(label_embeds[label_targets == i], dim=0)
        new_l_proto = self.l_moment * self.l_proto + (1 - self.l_moment) * new_l_proto
        self.set_protos(new_l_proto, self.u_proto)
