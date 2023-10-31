import argparse
import os
import sys
from argparse import ArgumentParser
from typing import Optional, Any, Dict

from torchvision.datasets.imagenet import ARCHIVE_META, ImageNet

from datasets.classification import BaseClassificationDataModule, ImageNetWithStats
from datasets.patch_sampler import RandomUniformSampler
from datasets.three_augment import three_augment
from datasets.utils import get_default_img_transform


class ElasticImageNetDataset(ImageNetWithStats):

    def __init__(self, root: str, patch_sampler=None, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)
        self.patch_sampler = patch_sampler

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample, target = super().__getitem__(index)

        if self.patch_sampler:
            patches, coords = self.patch_sampler(sample)
            return {
                "image": sample,
                "patches": patches,
                "coords": coords,
                "label": target
            }
        else:
            return {
                "image": sample,
                "label": target
            }


class ElasticImageNet1k(BaseClassificationDataModule):
    has_test_data = False
    cls_num_classes = 1000

    def __init__(self, *args, in_mem=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_mem = in_mem
        self.disk_dir = self.data_dir
        if self.in_mem:
            memfs_path = os.environ['MEMFS']
            print('using memfs at:', memfs_path, file=sys.stderr)
            self.data_dir = os.path.join(memfs_path, 'imagenet')
        self.prepare_data_per_node = True

        self.patch_sampler = RandomUniformSampler(random_patches=49, min_patch_size=16, max_patch_size=48)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(ElasticImageNet1k.__name__)
        parser.add_argument('--in-mem',
                            help='load dataset to ram',
                            type=bool,
                            default=False,
                            action=argparse.BooleanOptionalAction)
        return parent_parser

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == 'fit':
            self.train_dataset = ElasticImageNetDataset(root=self.data_dir, patch_sampler=self.patch_sampler,
                                                        split='train',
                                                        transform=
                                                        get_default_img_transform(self.image_size)
                                                        if self.force_no_augment else
                                                        three_augment(self.image_size))
            self.val_dataset = ElasticImageNetDataset(root=self.data_dir, patch_sampler=self.patch_sampler, split='val',
                                                      transform=get_default_img_transform(self.image_size))
        else:
            raise NotImplemented()

    def prepare_data(self) -> None:
        if self.in_mem:
            os.mkdir(self.data_dir)
            for tar_file, _ in ARCHIVE_META.values():
                os.symlink(os.path.join(self.disk_dir, tar_file), os.path.join(self.data_dir, tar_file))
            print('unpacking train set', file=sys.stderr)
            ImageNet(self.data_dir, split='train')
            print('unpacking val set', file=sys.stderr)
            ImageNet(self.data_dir, split='val')
            print('done unpacking imagenet at:', self.data_dir, file=sys.stderr)
