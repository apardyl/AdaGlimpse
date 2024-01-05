import numpy as np
from timm.data.mixup import mixup_target
from torch import Tensor

from architectures.rl.shared_memory import SharedMemory


class InPlacePatchMix:
    def __init__(self, patch_mix_alpha=1.0, prob=1.0, label_smoothing=0.0, num_classes=1000) -> None:
        self.patch_mix_alpha = patch_mix_alpha
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def __call__(self, shared_memory: SharedMemory) -> Tensor:
        patches: Tensor = shared_memory.current_patches
        coords: Tensor = shared_memory.current_coords
        targets: Tensor = shared_memory.target

        assert patches.shape[0] % 2 == 0, 'Batch size should be even when using this'

        lam = 1.
        if np.random.rand() < self.mix_prob:
            lam_mix = np.random.beta(self.patch_mix_alpha, self.patch_mix_alpha)
            lam = float(lam_mix)

        if lam != 1.:
            p = int(lam * patches.shape[1])  # patches to keep
            lam = p / patches.shape[1]  # round lambda to patch

            patches[:, p:, ...] = patches.flip(0)[:, p:, ...]
            coords[:, p:, ...] = coords.flip(0)[:, p:, ...]

        targets = mixup_target(targets, self.num_classes, lam, self.label_smoothing)
        return targets
