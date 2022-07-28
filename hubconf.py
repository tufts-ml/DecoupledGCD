import ccgaussian.load


dependencies = ["torch"]


def dino_ccg(pretrained=True, **kwargs):
    return ccgaussian.load.load_dino_ccg(pretrained, **kwargs)
