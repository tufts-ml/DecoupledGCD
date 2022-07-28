import torch

import ccgaussian.model


def load_dino_ccg(map_location=torch.device("cpu"), pretrained=True, **kwargs):
    num_classes = 5
    model = ccgaussian.model.DinoCCG(num_classes, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/tufts-ai-robotics-group/CCGaussian/releases/download/Model/DinoCCG_0_0_1.pt",  # noqa: E501
            map_location=map_location,
        )
        model.load_state_dict(state_dict)
    else:
        model = model.to(map_location)
    return model
