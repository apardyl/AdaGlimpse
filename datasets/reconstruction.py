import abc
import os.path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.base import BaseDataModule
from datasets.utils import get_default_aug_img_transform, get_default_img_transform


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, file_list=None, transform=None):
        assert root_dir is not None or file_list is not None
        if root_dir is not None:
            self.data = [os.path.join(root_dir, p) for p in os.listdir(root_dir)]
        else:
            self.data = file_list

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = Image.open(sample).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.zeros(1)  # empty


class BaseReconstructionDataModule(BaseDataModule, abc.ABC):
    pass


class Coco2014Reconstruction(BaseReconstructionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        train_dir = os.path.join(self.data_dir, 'train2014')
        val_dir = os.path.join(self.data_dir, 'val2014')
        test_dir = os.path.join(self.data_dir, 'test2014')

        if stage == 'fit':
            self.train_dataset = ReconstructionDataset(root_dir=train_dir,
                                                       transform=
                                                       get_default_img_transform(self.image_size)
                                                       if self.no_aug else
                                                       get_default_aug_img_transform(self.image_size))
            self.val_dataset = ReconstructionDataset(root_dir=val_dir,
                                                     transform=get_default_img_transform(self.image_size))
        elif stage == 'test':
            self.test_dataset = ReconstructionDataset(root_dir=test_dir,
                                                      transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class Sun360Reconstruction(BaseReconstructionDataModule):
    has_test_data = False

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'meta/sun360-dataset26-random.txt')) as f:
            file_list = f.readlines()
        file_list = [os.path.join(self.data_dir, p.strip()) for p in file_list]
        val_list = file_list[:len(file_list) // 10]
        train_list = file_list[len(file_list) // 10:]

        if stage == 'fit':
            self.train_dataset = ReconstructionDataset(file_list=train_list,
                                                       transform=
                                                       get_default_img_transform(self.image_size)
                                                       if self.no_aug else
                                                       get_default_aug_img_transform(self.image_size, scale=False))
            self.val_dataset = ReconstructionDataset(file_list=val_list,
                                                     transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class ADE20KReconstruction(BaseReconstructionDataModule):
    has_test_data = False

    def setup(self, stage: Optional[str] = None) -> None:
        train_dir = os.path.join(self.data_dir, 'train', 'images')
        val_dir = os.path.join(self.data_dir, 'val', 'images')

        if stage == 'fit':
            self.train_dataset = ReconstructionDataset(root_dir=train_dir,
                                                       transform=
                                                       get_default_img_transform(self.image_size)
                                                       if self.no_aug else
                                                       get_default_aug_img_transform(self.image_size))
            self.val_dataset = ReconstructionDataset(root_dir=val_dir,
                                                     transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()


class TestImageDirReconstruction(BaseReconstructionDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'test':
            self.test_dataset = ReconstructionDataset(root_dir=self.data_dir,
                                                      transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()
