import torch

import dpn.model


def load_dino_ccg(pretrained=True, **kwargs):
    num_classes = 5
    model = dpn.model.DinoCCG(num_classes, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/tufts-ai-robotics-group/CCGaussian/releases/download/NoveltyModel0.1.0/DinoCCG_0_1_0.pt",  # noqa: E501
            map_location="cpu",
        )
        model.load_state_dict(state_dict)
    return model


def load_ccg_gcd(pretrained=True, **kwargs):
    num_classes = 5
    model = dpn.model.DinoCCG(num_classes, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/tufts-ai-robotics-group/CCGaussian/releases/download/GCDModel/CCGGCD_0_0_1.pt",  # noqa: E501
            map_location="cpu",
        )
        model.load_state_dict(state_dict)
    return model
