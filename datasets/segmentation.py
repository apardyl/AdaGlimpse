import abc
import os
import torch

from PIL import Image
from torch.utils.data import DataLoader

from datasets.base import BaseDataModule
from datasets.segmentation_transforms import get_aug_seg_transforms, get_seg_transforms


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, classes_start, num_classes, transform=None):
        self.root_dir = root_dir
        self.classes_start = classes_start
        self.num_classes = num_classes
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, "images"))
        self.masks = os.listdir(os.path.join(root_dir, "masks"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_id = self.images[index]
        image = Image.open(os.path.join(self.root_dir, "images", image_id)).convert("RGB")
        mask = Image.open(os.path.join(self.root_dir, "masks", image_id.replace(".jpg", ".png")))

        image, mask = self.transform(image, mask)
        # Perform class mapping, 0 is unlabeled and is not counted as a class
        mask -= (self.classes_start - 1)
        mask = torch.clip(mask, 0, self.num_classes)

        # Turn unlabeled class 0 into 255, count valid classes from 0
        mask[mask == 0] = 256
        mask -= 1
        return image, mask


class BaseSegmentationDataModule(BaseDataModule, abc.ABC):
    num_classes = -1
    ignore_label = -1


class ADE20KSegmentation(BaseSegmentationDataModule):
    has_test_data = False
    classes_start = 1
    num_classes = 150
    ignore_label = 255

    def setup(self, stage="fit") -> None:
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        # test_dir = os.path.join(self.data_dir, 'test')

        if stage == 'fit':
            self.train_dataset = SegmentationDataset(train_dir, self.classes_start, self.num_classes,
                                                     transform=get_aug_seg_transforms(self.image_size))
            self.val_dataset = SegmentationDataset(val_dir, self.classes_start, self.num_classes,
                                                   transform=get_seg_transforms(self.image_size))
        elif stage == 'test':
            raise RuntimeError("There is no labeled test set for ADE20K")


class COCOStuffSegmentation(BaseSegmentationDataModule):
    has_test_data = False
    classes_start = 92
    num_classes = 92
    ignore_label = 255

    def setup(self, stage="fit") -> None:
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        # test_dir = os.path.join(self.data_dir, 'test')

        if stage == 'fit':
            self.train_dataset = SegmentationDataset(train_dir, self.classes_start, self.num_classes,
                                                     transform=get_aug_seg_transforms(self.image_size))
            self.val_dataset = SegmentationDataset(val_dir, self.classes_start, self.num_classes,
                                                   transform=get_seg_transforms(self.image_size))
        elif stage == 'test':
            raise RuntimeError("There is no labeled test set for COCOStuff")
