import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import label2rgb

from architectures.segmentation import SegmentationMae
from datasets.utils import IMAGENET_MEAN, IMAGENET_STD

_imagenet_mean = torch.tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(1)
_imagenet_std = torch.tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(1)

plt.rcParams.update({'figure.max_open_warning': 50})


def upscale_patch_values(mask, patch_size=(16, 16)):
    return mask.repeat(patch_size[1], axis=1).repeat(patch_size[0], axis=0)


def save_reconstructions(model, output, batch, vis_id, dump_path, last_only=False):
    mae = model.mae
    is_segmentation = isinstance(model, SegmentationMae)
    steps = output['steps']
    has_att_info = len(steps[3]['entropy']) > 0
    B = batch[0].shape[0]
    patch_size = mae.patch_embed.patch_size
    grid_h = mae.grid_size[0]
    grid_w = mae.grid_size[1]

    with torch.no_grad():
        outs = torch.stack([x['out'] for x in steps], dim=1).reshape(
            B * len(steps), *output['out'].shape[1:])
        outs = mae.unpatchify(outs)
        outs = outs.reshape(B, len(steps), *outs.shape[1:])

        if not is_segmentation:
            outs = torch.clip(
                (outs * _imagenet_std + _imagenet_mean), 0, 1).permute(0, 1, 3, 4, 2)

        inputs = torch.clip((batch[0].cpu() * _imagenet_std + _imagenet_mean), 0, 1).permute(0, 2, 3, 1).numpy()
        if is_segmentation:
            targets = batch[1].cpu()

        msk = torch.stack([x['mask'] for x in steps], dim=1)
        msk = msk.reshape(B, len(steps), grid_h, grid_w)

        # Dump visualizations
        os.makedirs(dump_path, exist_ok=True)
        for i in range(B):
            sub_path = os.path.join(dump_path, f'{vis_id}_{i}')
            os.makedirs(sub_path, exist_ok=True)
            for s in range(len(steps)):
                if last_only and s + 1 < len(steps):
                    continue

                msk_image = np.array(msk[i][s])
                out_image = np.array(outs[i][s])

                msk_upscales = upscale_patch_values(msk_image, patch_size)
                plt.imsave(os.path.join(sub_path, f'{s}_input.jpg'), inputs[i])
                plt.imsave(os.path.join(sub_path, f'{s}_mask.jpg'), msk_upscales)
                plt.imsave(os.path.join(sub_path, f'{s}_masked.jpg'),
                           upscale_patch_values(inputs[i] * msk_upscales[..., np.newaxis], patch_size))
                if not is_segmentation:
                    plt.imsave(os.path.join(sub_path, f'{s}_out.jpg'), out_image)
                else:
                    target = label2rgb(targets[i].numpy(), bg_label=255, bg_color=(0, 0, 0))
                    plt.imsave(os.path.join(sub_path, f'{s}_target.jpg'), target)
                    out = label2rgb(np.argmax(out_image, axis=-3), bg_label=255, bg_color=(0, 0, 0))
                    plt.imsave(os.path.join(sub_path, f'{s}_out.jpg'), out)

                if has_att_info:
                    if s + 1 < len(steps) and 'entropy' in steps[s + 1]:
                        plt.imsave(os.path.join(sub_path, f'{s}_entropy.jpg'),
                                   upscale_patch_values(steps[s + 1]['entropy'][i][0].cpu().numpy(), patch_size))
                        plt.imsave(os.path.join(sub_path, f'{s}_next.jpg'),
                                   upscale_patch_values(steps[s + 1]['selection_map'][i][0].cpu().numpy(), patch_size))
