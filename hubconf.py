import ccgaussian.load


dependencies = ["torch"]


def dino_ccg(pretrained=True, **kwargs):
    return ccgaussian.load.load_dino_ccg(pretrained, **kwargs)


def ccg_gcd(pretrained=True, **kwargs):
    return ccgaussian.load.load_ccg_gcd(pretrained, **kwargs)
