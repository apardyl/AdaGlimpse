from torchvision.transforms import RandomResizedCrop, InterpolationMode, RandomHorizontalFlip, Resize, ToTensor, \
    Normalize, Compose

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_default_aug_img_transform(img_size, scale=True):
    aug = []
    if scale:
        aug.append(RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC))
    aug += [
        RandomHorizontalFlip(),
        Resize(img_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return Compose(aug)


def get_default_img_transform(img_size):
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
