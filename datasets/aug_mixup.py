import numpy as np
import torch
from timm.data.mixup import mixup_target
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from datasets.patch_sampler import GridSampler, RandomUniformSampler, RandomMultiscaleSampler, RandomDelegatedSampler


class AugCollatePatchSampleMixup:
    def __init__(self, crop_sizes=None, random_patches=None, random_min_size=8, random_max_size=48,
                 grid_patch_size=None, grid_to_random_ratio=0.7, augment=True, mixup_alpha=1., cutmix_alpha=0.,
                 prob=1.0, switch_prob=0.5, correct_lam=True, label_smoothing=0.1, num_classes=1000) -> None:
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.correct_lam = correct_lam

        if grid_patch_size is not None:
            self.patch_sampler = GridSampler(patch_size=(grid_patch_size, grid_patch_size))
            print('using grid sampler')
            if random_patches is not None:
                random_patch_sampler = RandomUniformSampler(random_patches, random_min_size, random_max_size)
                print('using random sampler')
                self.patch_sampler = RandomDelegatedSampler([
                    (self.patch_sampler, grid_to_random_ratio),
                    (random_patch_sampler, 1 - grid_to_random_ratio)
                ])
        elif random_patches is not None:
            print('using random sampler')
            self.patch_sampler = RandomUniformSampler(random_patches, random_min_size, random_max_size)
        elif crop_sizes is not None:
            print('using random multiscale sampler')
            self.patch_sampler = RandomMultiscaleSampler(eval(crop_sizes))
        else:
            raise ValueError('no patch sampling parameters specified')

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mixup_batch(self, x, lam):
        if lam == 1.:
            return 1.
        x_flipped = x.flip(0).mul_(1. - lam)
        x.mul_(lam).add_(x_flipped)
        return lam

    def _patchmix_batch(self, x, coords, lam):
        if lam == 1.:
            return 1.
        p = int(lam * x.shape[1])  # patches to keep
        lam = p / x.shape[1]  # round lambda to patch
        x[:, p:, ...] = x.flip(0)[:, p:, ...]
        coords[:, p:, ...] = coords.flip(0)[:, p:, ...]
        return lam

    def on_epoch(self):
        if not self.augment:
            self.patch_sampler.reset_state()

    def __call__(self, batch):
        images, targets = collate(batch, collate_fn_map=default_collate_fn_map)
        assert images.shape[0] % 2 == 0, 'Batch size should be even when using this'

        lam, use_cutmix = self._params_per_batch()

        if self.augment and not use_cutmix:
            lam = self._mixup_batch(images, lam)

        patches = []
        coords = []
        for im in images:
            pt, cs = self.patch_sampler(im)
            patches.append(pt)
            coords.append(cs)
        images = torch.stack(patches, dim=0)
        del patches
        coords = torch.stack(coords, dim=0)

        if self.augment and use_cutmix:
            lam = self._patchmix_batch(images, coords, lam)

        if self.augment:
            targets = mixup_target(targets, self.num_classes, lam, self.label_smoothing)
        return (images, coords), targets
