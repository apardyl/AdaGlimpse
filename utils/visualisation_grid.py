from typing import Optional, List

import torch
from matplotlib import pyplot as plt
from torch import Tensor

from architectures.rl_glimpse import BaseRlMAE
from utils.visualisation_blocks import GridField, ImageGridField, SegmentationGridField, ClsPredictionGridField, \
    ClsTargetGridField, ScoreGridField, Coords, bbox_map, glimpse_map


def show_grid(grid: List[List[Optional[GridField]]], name):
    rows = len(grid[0])
    columns = len(grid)
    size = 3
    height_ratios = [row.size_ratio for row in grid[-1]]
    fig, axs = plt.subplots(
        figsize=(columns * size, sum(height_ratios)), ncols=columns, nrows=rows, squeeze=False,
        height_ratios=height_ratios
    )
    for x, col in enumerate(grid):
        for y, field in enumerate(col):
            if field is not None:
                field.render(axs[y, x])
            else:
                axs[y, x].set_axis_off()
    plt.tight_layout()
    plt.savefig(name)
    plt.close(fig)


def save_grid(grid: List[List[Optional[GridField]]], path):
    for x, col in enumerate(grid):
        for y, field in enumerate(col):
            if field is not None and (isinstance(field, ImageGridField) or isinstance(field, SegmentationGridField)):
                field.save(path.replace('.png', f'_{x}_{y}.png'))


def visualize_grid(model: BaseRlMAE, image: Tensor, out: Tensor, coords: Tensor, patches: Tensor, scores: Tensor,
                   target: Tensor, done: Tensor, save_path: str, rev_normalizer, save_all: bool) -> None:
    num_glimpses = model.num_glimpses
    patches_per_glimpse = coords.shape[0] // num_glimpses
    assert patches_per_glimpse * num_glimpses == coords.shape[0]  # assert if divisible by num_glimpses

    if len(out.shape) == 3:
        # reconstruction
        out = model.mae.unpatchify(out)
        out = rev_normalizer(out).to(torch.uint8)

        pred_field = ImageGridField
        target_field = None
    elif len(out.shape) == 2:
        # classification
        pred_field = ClsPredictionGridField
        target_field = ClsTargetGridField
    elif len(out.shape) == 4:
        pred_field = SegmentationGridField
        target_field = SegmentationGridField

    else:
        raise NotImplementedError()

    grid: List[List[Optional[GridField]]] = [
        [
            ImageGridField(image),
            None,
            pred_field(out[0])
        ]
        + ([target_field(target)] if target_field is not None else [])
        + [ScoreGridField(scores[0])]
    ]

    coords = [Coords.from_tensor(x) for x in coords]
    merged_coords = [Coords.merge(coords[idx * patches_per_glimpse: idx * patches_per_glimpse + patches_per_glimpse])
                     for idx in range(num_glimpses)]

    for glimpse_idx in range(model.num_glimpses):
        if done[glimpse_idx]:
            continue
        start_idx = glimpse_idx * patches_per_glimpse
        end_idx = start_idx + patches_per_glimpse

        grid.append(
            [
                ImageGridField(bbox_map(image, coords[:end_idx], merged_coords[:glimpse_idx + 1])),
                ImageGridField(glimpse_map(patches, coords[:end_idx], image.shape)),
                pred_field(out[glimpse_idx + 1])
            ]
            + ([target_field(target)] if target_field is not None else [])
            + [ScoreGridField(scores[glimpse_idx + 1])]
        )

    show_grid(grid, save_path)
    if save_all:
        save_grid(grid, save_path)
