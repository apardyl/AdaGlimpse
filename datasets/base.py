import abc
import argparse
import os
import sys
from argparse import ArgumentParser

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import RandomSampler, DataLoader


class BaseDataModule(LightningDataModule, abc.ABC):
    has_test_data = True

    def __init__(self, data_dir, train_batch_size=32, eval_batch_size=32, num_workers=8, num_samples=None,
                 image_size=(224, 224), force_no_augment=False, mem_fs=False, always_drop_last=False, **_):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.image_size = image_size
        self.force_no_augment = force_no_augment
        self.always_drop_last = always_drop_last

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.mem_fs = mem_fs
        self.disk_dir = self.data_dir
        if self.mem_fs:
            memfs_path = os.environ['MEMFS']
            print('using memfs at:', memfs_path, file=sys.stderr)
            self.data_dir = os.path.join(memfs_path, 'dataset')
            self.prepare_data_per_node = True

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--data-dir',
                            help='dataset location',
                            type=str,
                            required=True)
        parser.add_argument('--train-batch-size',
                            help='batch size for training',
                            type=int,
                            default=32)
        parser.add_argument('--eval-batch-size',
                            help='batch size for validation and testing',
                            type=int,
                            default=32)
        parser.add_argument('--num-workers',
                            help='dataloader workers per DDP process',
                            type=int,
                            default=8)
        parser.add_argument('--num-samples',
                            help='number of images to sample in each training epoch',
                            type=int,
                            default=None)
        parser.add_argument('--image-size',
                            help='image size H W',
                            type=int,
                            nargs=2,
                            default=(224, 224))
        parser.add_argument('--mem-fs',
                            help='load dataset to MEMFS',
                            type=bool,
                            default=False,
                            action=argparse.BooleanOptionalAction)
        return parent_parser

    def train_dataloader(self, sampler=None) -> TRAIN_DATALOADERS:
        print(f'Loaded {len(self.train_dataset)} train samples', file=sys.stderr)
        if self.num_samples is not None:
            sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=self.num_samples)
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
                          sampler=sampler, shuffle=None if sampler is not None else True, drop_last=True,
                          pin_memory=True)

    def test_dataloader(self, sampler=None) -> EVAL_DATALOADERS:
        print(f'Loaded {len(self.test_dataset)} test samples', file=sys.stderr)
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, sampler=sampler, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=self.always_drop_last)

    def val_dataloader(self, sampler=None) -> EVAL_DATALOADERS:
        print(f'Loaded {len(self.val_dataset)} val samples', file=sys.stderr)
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, sampler=sampler, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=self.always_drop_last)

    def _load_to_memfs(self):
        raise NotImplemented()

    def prepare_data(self) -> None:
        if self.mem_fs:
            self._load_to_memfs()
