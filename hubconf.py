import torch

import ccgaussian.load


dependencies = ["torch"]


def dino_ccg(map_location=torch.device("cpu"), pretrained=True, **kwargs):
    return ccgaussian.load.load_dino_ccg(map_location, pretrained, **kwargs)
