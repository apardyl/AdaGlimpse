from typing import Optional, Any, Dict

from datasets.classification import ImageNetWithStats, ImageNet1k
from datasets.patch_sampler import RandomUniformSampler
from datasets.three_augment import three_augment
from datasets.utils import get_default_img_transform


class ElasticImageNetDataset(ImageNetWithStats):

    def __init__(self, root: str, patch_sampler=None, split: str = "train", **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)
        assert patch_sampler is not None
        self.patch_sampler = patch_sampler

    def __getitem__(self, index: int) -> Dict[str, Any]:
        batch = super().__getitem__(index)
        patches, coords = self.patch_sampler(batch['image'])
        return batch | {
            "patches": patches,
            "coords": coords,
        }


class ElasticImageNet1k(ImageNet1k):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_sampler = RandomUniformSampler(random_patches=49, min_patch_size=16, max_patch_size=48)

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == 'fit' or stage == 'validate':
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
