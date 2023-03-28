from torchvision import transforms


INTERP = transforms.InterpolationMode.BICUBIC
IMAGENET_SHAPE = (3, 224, 224)
IMAGE_SIZE = IMAGENET_SHAPE[1]


def dino_norm_trans():
    # DINO normalization for ImageNet pretraining
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def byol_train():
    # BYOL symmetric augmentation from Appendix F.3
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(23)], p=0.5),
        transforms.RandomSolarize(.5, p=.2),
        transforms.RandomHorizontalFlip(),
        dino_norm_trans(),
    ])


def sim_gcd_train():
    # SimGCD training transform
    # uses random ColorJitter, while original implementation effectively didn't use it
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=INTERP),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        dino_norm_trans(),
    ])


def sim_gcd_test():
    # SimGCD testing transform
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=INTERP),
        transforms.CenterCrop(IMAGE_SIZE),
        dino_norm_trans(),
    ])
