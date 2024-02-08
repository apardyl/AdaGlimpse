import torch
from torch import Tensor
from torchrl.objectives import SACLoss


class GlimpseSACLoss(SACLoss):
    def __init__(self, num_glimpses: int, *args, popart=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.popart = popart
        self.num_glimpses = num_glimpses

        if self.popart:
            self.register_buffer('running_mean', torch.zeros(self.num_glimpses))
            self.register_buffer('running_var', torch.ones(self.num_glimpses))
            self.beta1, self.beta2 = 0.01, 0.01

    def _popart(self, target: Tensor, steps: Tensor) -> Tensor:
        assert len(target.shape) == 1

        value_sum = torch.zeros(self.num_glimpses, device=target.device)
        value_2_sum = torch.zeros(self.num_glimpses, device=target.device)
        count = torch.zeros(self.num_glimpses, device=target.device, dtype=torch.long)

        for glimpse_idx in range(self.num_glimpses):
            mask = (steps == glimpse_idx).reshape(target.shape[0])
            masked_target = target[mask]
            value_sum[glimpse_idx] = masked_target.sum().detach()
            value_2_sum[glimpse_idx] = masked_target.pow(2).sum().detach()
            count[glimpse_idx] = mask.sum().detach()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(value_sum)
            torch.distributed.all_reduce(value_2_sum)
            torch.distributed.all_reduce(count)

        count_clip = torch.clip(count, 1)
        value_mean = value_sum / count_clip
        value_2_mean = value_2_sum / count_clip
        scale = count / target.shape[0]
        self.running_mean = self.running_mean * (1 - self.beta1 * scale) + (self.beta1 * scale) * value_mean
        self.running_var = self.running_var * (1 - self.beta2 * scale) + (self.beta2 * scale) * value_2_mean
        sigma = (self.running_var - self.running_mean ** 2) ** 0.5

        for glimpse_idx in range(self.num_glimpses):
            mask = (steps == glimpse_idx).reshape(target.shape[0])
            target[mask] = (target[mask] - self.running_mean[glimpse_idx]) / sigma[glimpse_idx]

        return target

    def _compute_target_v2(self, tensordict) -> Tensor:
        target = super()._compute_target_v2(tensordict)

        if self.popart:
            target = self._popart(target, tensordict['next', 'step'])

        return target
