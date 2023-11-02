import abc
import os
import sys
from collections import Counter
from typing import Optional, Tuple, Any, Dict

import torch
from PIL import Image
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import RandomSampler
from torchvision.datasets import ImageNet
from torchvision.datasets.imagenet import ARCHIVE_META

from datasets.base import BaseDataModule
from datasets.three_augment import three_augment
from datasets.utils import get_default_img_transform, get_default_aug_img_transform


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.data = file_list
        self.labels = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(sample).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return {
            'image': sample,
            'label': self.labels[index]
        }

    def class_stats(self):
        return [v for k, v in sorted(Counter(self.labels).items())]


class BaseClassificationDataModule(BaseDataModule, abc.ABC):
    cls_num_classes = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inst_num_classes = None

    @property
    def num_classes(self):
        if self.inst_num_classes is not None:
            return self.inst_num_classes
        return self.cls_num_classes

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # print(f'Train class statistics:', self.train_dataset.class_stats(), file=sys.stderr)
        return super().train_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # print(f'Test class statistics:', self.test_dataset.class_stats(), file=sys.stderr)
        return super().test_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # print(f'Val class statistics:', self.val_dataset.class_stats(), file=sys.stderr)
        return super().val_dataloader()


class Sun360Classification(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 26

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'meta/sun360-dataset26-random.txt')) as f:
            file_list = f.readlines()
        labels = [str.join('/', p.split('/')[:2]) for p in file_list]
        classes = {name: idx for idx, name in enumerate(sorted(set(labels)))}
        labels = [classes[x] for x in labels]
        file_list = [os.path.join(self.data_dir, p.strip()) for p in file_list]
        val_list = file_list[:len(file_list) // 10]
        val_labels = labels[:len(file_list) // 10]
        train_list = file_list[len(file_list) // 10:]
        train_labels = labels[len(file_list) // 10:]

        if stage == 'fit':
            self.train_dataset = ClassificationDataset(file_list=train_list, label_list=train_labels,
                                                       transform=
                                                       get_default_img_transform(self.image_size)
                                                       if self.force_no_augment else
                                                       get_default_aug_img_transform(self.image_size, scale=False))
            self.val_dataset = ClassificationDataset(file_list=val_list, label_list=val_labels,
                                                     transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class ImageNetWithStats(ImageNet):
    def class_stats(self):
        return [v for k, v in sorted(Counter(self.targets).items())]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample, target = super().__getitem__(index)
        return {
            "image": sample,
            "label": target
        }


class ImageNet1k(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 1000

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == 'fit' or stage == 'validate':
            self.train_dataset = ImageNetWithStats(root=self.data_dir, split='train',
                                                   transform=
                                                   get_default_img_transform(self.image_size)
                                                   if self.force_no_augment else
                                                   three_augment(self.image_size))
            self.val_dataset = ImageNetWithStats(root=self.data_dir, split='val',
                                                 transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()

    def _load_to_memfs(self) -> None:
        os.mkdir(self.data_dir)
        for tar_file, _ in ARCHIVE_META.values():
            os.symlink(os.path.join(self.disk_dir, tar_file), os.path.join(self.data_dir, tar_file))
        print('unpacking train set', file=sys.stderr)
        ImageNet(self.data_dir, split='train')
        print('unpacking val set', file=sys.stderr)
        ImageNet(self.data_dir, split='val')
        print('done unpacking imagenet at:', self.data_dir, file=sys.stderr)
