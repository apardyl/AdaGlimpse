import argparse
import os.path
import random
import sys

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.prepare import experiment_from_args

random.seed(1)
torch.manual_seed(1)
torch.set_float32_matmul_precision("high")


def define_args(parent_parser):
    parser = parent_parser.add_argument_group("predict.py")
    parser.add_argument(
        "--max-batches",
        help="number of batches from dataset to process",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--batch-size", '--bs',
        help="size of batches",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--visualization-path",
        help="path to save visualizations to",
        type=str,
        default="predict_outputs",
    )
    return parent_parser


class UserHook:
    def __init__(self):
        self.images = []
        self.latent = []
        self.out = []
        self.coords = []
        self.patches = []
        self.selection_mask = []

    def _stack_last_step(self):
        self.out[-1] = torch.stack(self.out[-1], dim=1)
        self.coords[-1] = torch.stack(self.coords[-1], dim=1)
        self.patches[-1] = torch.stack(self.patches[-1], dim=1)
        self.selection_mask[-1] = torch.stack(self.selection_mask[-1], dim=1)

    def __call__(self, step, out, next_glimpse, image, selection_mask, patches):
        if step == 0:
            self.images.append(image)
            if len(self.out) > 0:
                # stack by the step dimension
                self._stack_last_step()
            self.out.append([])
            self.coords.append([])
            self.patches.append([])
            self.selection_mask.append([])
        # obtain number of glimpses done in single step
        num_of_glimpses = next_glimpse.shape[1]
        self.out[-1].append(out)
        self.coords[-1].append(next_glimpse)
        self.patches[-1].append(patches[:, -num_of_glimpses:, :, :, :])
        self.selection_mask[-1].append(selection_mask)

    def stack(self):
        if type(self.out[-1]) is list:
            self._stack_last_step()
        self.images = torch.cat(self.images, dim=0)
        self.out = torch.cat(self.out, dim=0)
        self.coords = torch.cat(self.coords, dim=0)
        self.patches = torch.cat(self.patches, dim=0)
        self.selection_mask = torch.cat(self.selection_mask, dim=0)


def show_grid(imgs, name=None):
    rows = len(imgs[0])
    columns = len(imgs)
    size = round(imgs[0][0].shape[-1] / 100 + 1, 0)
    fig, axs = plt.subplots(
        figsize=(columns * size, rows * size), ncols=columns, nrows=rows, squeeze=False
    )
    for y, row in enumerate(imgs):
        for x, img in enumerate(row):
            while len(img.shape) > 3:
                img = img.squeeze(0)
            if len(img.shape) < 3:
                img = img.unsqueeze(0)
            img = img.detach().float()
            img = torchvision.transforms.functional.to_pil_image(img)
            axs[x, y].imshow(np.asarray(img), resample=False)
            axs[x, y].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    if name:
        plt.savefig(name)
    else:
        plt.show()


def combine_patches(patch_set, glimpse_sets, coords, output_shape):
    img = torch.zeros(output_shape)
    for glimpse, patch in zip(glimpse_sets, patch_set):
        x, y, z = glimpse
        img[:, x: x + z, y: y + z] = torchvision.transforms.functional.resize(
            patch,
            (z, z),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
    return img[:, coords[0]: coords[2], coords[1]: coords[3]]


def glimpse_map(patches, glimpses, output_shape):
    img = torch.zeros(output_shape)
    glimpse_coords = [combine_glimpses(glimpse_set) for glimpse_set in glimpses]
    for _, coords, patch_set, glimpse_set in sorted([
        # sort by decreasing area of glimpse
        (-glimps_area(coords), coords, patch_set, glimpse_sets)
        for patch_set, coords, glimpse_sets in zip(patches, glimpse_coords, glimpses)
    ]):
        patch = combine_patches(patch_set, glimpse_set, coords, output_shape)
        img[:, coords[0]: coords[2], coords[1]: coords[3]] = patch
    return img


def combine_glimpses(glimpses, flip_xy=False):
    glimpses = glimpses.squeeze()
    for glimpse in glimpses.clone():
        x, y, z = glimpse
        glimpses = torch.cat((glimpses, torch.Tensor([[x + z, y + z, z]])), dim=0)
    if flip_xy:
        out = [
            glimpses[:, 1].min(),
            glimpses[:, 0].min(),
            glimpses[:, 1].max(),
            glimpses[:, 0].max(),
        ]
    else:
        out = [
            glimpses[:, 0].min(),
            glimpses[:, 1].min(),
            glimpses[:, 0].max(),
            glimpses[:, 1].max(),
        ]
    out = [int(x) for x in out]
    return out


def glimps_area(coords):
    return (coords[2] - coords[0]) * (coords[3] - coords[1])


def bbox_map(img, glimpses):
    boxes = torch.tensor(
        [combine_glimpses(glimpse_set, flip_xy=True) for glimpse_set in glimpses],
        dtype=torch.float,
    )
    colors = ["red" for _ in glimpses]
    img = (img * 255).clone().to(torch.uint8)  # required by draw_bounding_boxes
    return torchvision.utils.draw_bounding_boxes(img, boxes, colors=colors, width=3) / 255


def selection_map(mask, patch_size):
    while len(mask.shape) > 2:
        mask = mask.squeeze(0)
    mask = mask.detach().cpu().numpy()
    mask = mask.repeat(patch_size[1], axis=1).repeat(patch_size[0], axis=0)
    return torch.from_numpy(mask).unsqueeze(0)


def visualize(args, model):
    hook = model.user_forward_hook
    rounds = hook.coords.shape[1]
    patch_size = [x // y for x, y in zip(args.image_size, model.mae.grid_size)]
    for idx, (img, outs, coords, patches, selection_masks) in enumerate(
            zip(hook.images, hook.out, hook.coords, hook.patches, hook.selection_mask)
    ):
        zero_img = torch.zeros_like(img)
        images = [[0 for _ in range(4)] for _ in range(rounds + 1)]
        # 0th round
        images[0] = [img, zero_img, zero_img, zero_img]
        for glimpse_idx, (out, selection_mask) in enumerate(zip(outs, selection_masks)):
            glimpse_idx += 1
            coord, patch = [x[:glimpse_idx] for x in [coords, patches]]
            # 1st row, selection boxes
            images[glimpse_idx][0] = bbox_map(img, coord)
            # 2nd row, glimpses
            images[glimpse_idx][1] = glimpse_map(patch, coord, img.shape)
            # 3rd row, reconstruction
            images[glimpse_idx][2] = model.mae.unpatchify(out.unsqueeze(0)).clip(0, 1)
            # 4th row, selection mask
            images[glimpse_idx][3] = selection_map(selection_mask, patch_size)
        show_grid(images, os.path.join(args.visualization_path, f"{idx}.png"))


def main():
    data_module, model, args = experiment_from_args(
        sys.argv, add_argparse_args_fn=define_args
    )
    model.eval()
    model.user_forward_hook = UserHook()
    augmentation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.image_size),
            transforms.Lambda(lambda x: {"image": x}),
        ]
    )
    images = torchvision.datasets.ImageFolder(args.data_dir, transform=augmentation)
    images_loader = DataLoader(images, batch_size=args.batch_size, shuffle=False)
    for idx, (image, cls) in enumerate(
            tqdm(images_loader, total=min(len(images_loader), args.max_batches))
    ):
        if idx >= args.max_batches:
            break
        _ = model(image)
    model.user_forward_hook.stack()
    visualize(args, model)


if __name__ == "__main__":
    main()
