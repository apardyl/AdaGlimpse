import abc
import os
from glob import glob

import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import pil_to_tensor

from datasets.base import BaseDataModule
from datasets.segmentation_transforms import get_aug_seg_transforms, get_seg_transforms


class ADESegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, classes_start, num_classes, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.classes_start = classes_start
        self.num_classes = num_classes
        self.transform = transform
        self.images_files = os.listdir(os.path.join(root_dir, "images", split))

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        # noinspection PyTypeChecker
        image_name: str = self.images_files[index]
        image = Image.open(os.path.join(self.root_dir, "images", self.split, image_name)).convert("RGB")
        mask = Image.open(os.path.join(self.root_dir, "annotations", self.split, image_name.replace(".jpg", ".png")))

        image, mask = self.transform(image, mask)
        # Perform class mapping, 0 is unlabeled and is not counted as a class
        mask -= (self.classes_start - 1)
        mask = torch.clip(mask, 0, self.num_classes)

        # Turn unlabeled class 0 into 255, count valid classes from 0
        mask[mask == 0] = 256
        mask -= 1
        return {"image": image, "mask": mask}


class BaseSegmentationDataModule(BaseDataModule, abc.ABC):
    num_classes = -1
    ignore_label = -1


class ADE20KSegmentation(BaseSegmentationDataModule):
    has_test_data = False
    classes_start = 1
    num_classes = 150
    ignore_label = 255

    def setup(self, stage="fit") -> None:
        if stage == 'fit':
            self.train_dataset = ADESegmentationDataset(self.data_dir, "training", self.classes_start, self.num_classes,
                                                        transform=get_aug_seg_transforms(self.image_size))
            self.val_dataset = ADESegmentationDataset(self.data_dir, "validation", self.classes_start, self.num_classes,
                                                      transform=get_seg_transforms(self.image_size))
        elif stage == 'test':
            self.test_dataset = ADESegmentationDataset(self.data_dir, "validation", self.classes_start,
                                                       self.num_classes,
                                                       transform=get_seg_transforms(self.image_size))
