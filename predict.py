import os.path
import random
import sys
from dataclasses import dataclass
from operator import itemgetter
from typing import Optional, List

import torch
import torchvision.datasets
from lightning import Trainer
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm

from architectures.base import AutoconfigLightningModule
from architectures.rl.shared_memory import SharedMemory
from architectures.rl_glimpse import BaseRlMAE
from architectures.utils import RevNormalizer
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
        default="visualizations",
    )
    parser.add_argument(
        "--model-checkpoint",
        help='path to a saved model state',
        type=str,
        required=True
    )
    return parent_parser


class RLUserHook:
    def __init__(self):
        self.images = []
        self.latent = []
        self.out = []
        self.scores = []
        self.coords = []
        self.patches = []
        self.current_out = []
        self.current_scores = []

    def __call__(self, env_state: SharedMemory, out, score):
        self.current_out.append(out.clone().detach().cpu())
        self.current_scores.append(score.clone().detach().cpu())

        if env_state.is_done:
            self.images.append(env_state.images.clone().detach().cpu())
            self.coords.append(env_state.coords.clone().detach().cpu())
            self.patches.append(env_state.patches.clone().detach().cpu())
            self.out.append(torch.stack(self.current_out, dim=1))
            self.current_out = []
            self.scores.append(torch.stack(self.current_scores, dim=1))
            self.current_scores = []

    def compute(self):
        return {
            "images": torch.cat(self.images, dim=0),
            "out": torch.cat(self.out, dim=0),
            "scores": torch.cat(self.scores, dim=0),
            "coords": torch.cat(self.coords, dim=0),
            "patches": torch.cat(self.patches, dim=0)
        }


@dataclass
class Coords:
    y1: int
    x1: int
    y2: int
    x2: int

    @classmethod
    def from_tensor(cls, x):
        x = x.squeeze()
        assert len(x.shape) == 1 and x.shape[0] == 4
        return cls(int(x[0]), int(x[1]), int(x[2]), int(x[3]))

    def bbox(self):
        return self.x1, self.y1, self.x2, self.y2

    @classmethod
    def merge(cls, coords: List['Coords']):
        return Coords(
            min(c.y1 for c in coords),
            min(c.x1 for c in coords),
            max(c.y2 for c in coords),
            max(c.x2 for c in coords)
        )

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __lt__(self, other):
        return self.area() < other.area()


def show_grid(grid, name):
    rows = len(grid[0])
    columns = len(grid)
    size = round(grid[0][0].shape[-1] / 100 + 1, 0)
    fig, axs = plt.subplots(
        figsize=(columns * size, rows * size), ncols=columns, nrows=rows, squeeze=False,
        height_ratios=([5] * (rows - 1) + [1])
    )
    for y, row in enumerate(grid):
        for x, field in enumerate(row):
            if field is not None:
                if len(field.shape) == 1 and field.shape[0] == 1:
                    axs[x, y].set_axis_off()
                    axs[x, y].text(0.15, 0.4, f'Score: {float(field):.3f}',
                                   font={'size': 18})
                else:
                    if len(field.shape) == 3:
                        if field.shape[0] == 3:
                            field = field.permute((1, 2, 0))
                    axs[x, y].imshow(field, resample=False)
                    axs[x, y].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            else:
                axs[x, y].set_axis_off()
    plt.tight_layout()
    plt.savefig(name)


def glimpse_map(patches, coords: List[Coords], output_shape):
    img = torch.zeros(output_shape, dtype=torch.uint8)
    for coord, patch in sorted([(coord, patch) for patch, coord in zip(patches, coords)], key=itemgetter(0),
                               reverse=True):
        coord: Coords
        patch = resize(
            patch, [coord.y2 - coord.y1, coord.x2 - coord.x1],
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        img[:, coord.y1: coord.y2, coord.x1: coord.x2] = patch
    return img


def bbox_map(img, coords, merged_coords):
    boxes = torch.tensor(
        [c.bbox() for c in coords],
        dtype=torch.float,
    )
    merged_boxes = torch.tensor(
        [c.bbox() for c in merged_coords],
        dtype=torch.float,
    )
    colors = ["red"] * len(coords)
    merged_colors = ["red"] * len(merged_coords)
    img = torchvision.utils.draw_bounding_boxes(img, boxes, colors=colors, width=1)
    img = torchvision.utils.draw_bounding_boxes(img, merged_boxes, colors=merged_colors, width=3)
    return img


def selection_map(mask, patch_size):
    while len(mask.shape) > 2:
        mask = mask.squeeze(0)
    mask = mask.detach().cpu().numpy()
    mask = mask.repeat(patch_size[1], axis=1).repeat(patch_size[0], axis=0)
    return torch.from_numpy(mask).unsqueeze(0)


def visualize_one(model: BaseRlMAE, image: Tensor, out: Tensor, coords: Tensor, patches: Tensor, scores: Tensor,
                  save_path: str, rev_normalizer) -> None:
    num_glimpses = model.num_glimpses
    patches_per_glimpse = coords.shape[0] // num_glimpses
    assert patches_per_glimpse * num_glimpses == coords.shape[0]  # assert if divisible by num_glimpses

    out = model.mae.unpatchify(out)
    out = rev_normalizer(out).to(torch.uint8)

    grid: List[List[Optional[Tensor]]] = [
        [image, None, out[0], scores[0]]
    ]

    coords = [Coords.from_tensor(x) for x in coords]
    merged_coords = [Coords.merge(coords[idx * patches_per_glimpse: idx * patches_per_glimpse + patches_per_glimpse])
                     for idx in range(num_glimpses)]

    for glimpse_idx in range(model.num_glimpses):
        start_idx = glimpse_idx * patches_per_glimpse
        end_idx = start_idx + patches_per_glimpse

        grid.append([
            bbox_map(image, coords[:end_idx], merged_coords[:glimpse_idx + 1]),
            glimpse_map(patches, coords[:end_idx], image.shape),
            out[glimpse_idx + 1],
            scores[glimpse_idx + 1]
        ])

    show_grid(grid, save_path)


def visualize(visualization_path, model):
    data = model.user_forward_hook.compute()

    rev_normalizer = RevNormalizer()
    images = rev_normalizer(data["images"]).to(torch.uint8)
    patches = rev_normalizer(data["patches"]).to(torch.uint8)

    for idx, (img, out, coord, patch, score) in enumerate(tqdm(
            zip(images, data["out"], data["coords"], patches, data['scores']),
            total=images.shape[0])):
        visualize_one(model, img, out, coord, patch, score,
                      os.path.join(visualization_path, f"{idx}.png"), rev_normalizer)


def main():
    model: AutoconfigLightningModule
    data_module, model, args = experiment_from_args(
        sys.argv, add_argparse_args_fn=define_args
    )

    model.load_pretrained(args.model_checkpoint)

    visualization_path = args.visualization_path
    os.makedirs(visualization_path, exist_ok=True)

    model.eval()

    if isinstance(model, BaseRlMAE):
        model.user_forward_hook = RLUserHook()
    else:
        raise RuntimeError(f"Unrecognized model type: {type(model)}")

    trainer = Trainer()
    trainer.test(model)

    visualize(visualization_path, model)


if __name__ == "__main__":
    main()
