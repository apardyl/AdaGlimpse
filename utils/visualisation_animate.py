from enum import IntEnum

import torch
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Rectangle
from torch import Tensor
from torchvision.transforms.functional import resize, InterpolationMode

from architectures.rl_glimpse import BaseRlMAE
from utils.visualisation_blocks import Coords


def get_glimpse(image: Tensor, glimpse_coords: Coords, glimpse_grid_size: int) -> Tensor:
    patch = image[:, glimpse_coords.y1: glimpse_coords.y2, glimpse_coords.x1: glimpse_coords.x2]
    patch = resize(patch, [glimpse_grid_size * 16, glimpse_grid_size * 16], interpolation=InterpolationMode.BILINEAR)
    return patch


class FrameEnum(IntEnum):
    IMAGE = 0
    GLIMPSE = 1
    PATCHES = 2
    PREDICTION = 3
    ACTION = 4


def draw_frame(ax, stage: FrameEnum, image, glimpse_coords, glimpse_grid_size, known_coords, out, next_action):
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 320)
    plt.axis('off')

    x_pos = 1
    y_pos = 60

    ax.imshow(image.permute((1, 2, 0)), extent=(x_pos, x_pos + 224, y_pos, y_pos + 224))
    ax.text(x_pos + 10, y_pos + 235, "Unobserved scene (224x224)", fontsize=16)

    ax.imshow(plt.imread('utils/assets/videocam.png'), extent=(x_pos + 260, x_pos + 320, y_pos + 82, y_pos + 142))
    ax.text(x_pos + 270, y_pos + 150, "Glimpse\ncapture", fontsize=16)

    camera_lens = (x_pos + 265, y_pos + 112)
    if glimpse_coords is not None and stage >= FrameEnum.GLIMPSE:
        rect = Rectangle((glimpse_coords.x1 + x_pos, 224 - glimpse_coords.y2 + y_pos),
                         glimpse_coords.x2 - glimpse_coords.x1,
                         glimpse_coords.y2 - glimpse_coords.y1, linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        glimpse_corners = [
            (glimpse_coords.x1 + x_pos, 224 - glimpse_coords.y1 + y_pos),
            (glimpse_coords.x1 + x_pos, 224 - glimpse_coords.y2 + y_pos),
            (glimpse_coords.x2 + x_pos, 224 - glimpse_coords.y1 + y_pos),
            (glimpse_coords.x2 + x_pos, 224 - glimpse_coords.y2 + y_pos),
        ]

        for corner in glimpse_corners:
            ax.plot([camera_lens[0], corner[0]], [camera_lens[1], corner[1]], 'r-', lw=1)

    x_pos += 320

    if glimpse_coords is not None and stage >= FrameEnum.PATCHES:
        ax.arrow(x_pos, y_pos + 112, 25, 0, length_includes_head=True, width=1, head_width=5)

    x_pos += 35

    if glimpse_coords is not None and stage >= FrameEnum.PATCHES:
        patch = get_glimpse(image, glimpse_coords, glimpse_grid_size)
        ax.imshow(patch.permute((1, 2, 0)), extent=(x_pos, x_pos + 32, y_pos + 96, y_pos + 128),
                  interpolation='nearest')
        ax.text(x_pos - 8, y_pos + 150, f"({16 * glimpse_grid_size}x{16 * glimpse_grid_size})", fontsize=16)

    x_pos += 42

    if glimpse_coords is not None and stage >= FrameEnum.PATCHES:
        ax.arrow(x_pos, y_pos + 112, 25, 0, length_includes_head=True, width=1, head_width=5)

    x_pos += 35

    rect = Rectangle((x_pos, y_pos), 20,
                     224, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(x_pos - 40, y_pos + 235, "Known glimpses", fontsize=16)

    max_glimpses = 14
    assert len(known_coords) <= max_glimpses
    for idx, coord in enumerate(known_coords):
        if idx == len(known_coords) - 1 and stage < FrameEnum.PATCHES:
            break
        y = y_pos + ((max_glimpses - 1) * 16) - idx * 16
        patch = get_glimpse(image, coord, glimpse_grid_size)
        ax.imshow(patch.permute((1, 2, 0)), extent=(x_pos + 3, x_pos + 3 + 14, y, y + 14),
                  interpolation='nearest')

    x_pos += 30

    if stage >= FrameEnum.PREDICTION:
        ax.arrow(x_pos, y_pos + 112, 25, 0, length_includes_head=True, width=1, head_width=5)

    x_pos += 35

    ax.imshow(plt.imread('utils/assets/block.png'), extent=(x_pos, x_pos + 138, y_pos + 34, y_pos + 187))

    action = None
    if stage == FrameEnum.ACTION:
        action = next_action[-1]
    elif out.shape[0] == 2 and stage in {FrameEnum.IMAGE, FrameEnum.GLIMPSE}:
        action = next_action[0]

    if action is not None:
        x, y, z = action.x1, action.y1, action.x2 - action.x1
        x, y, z = x / 224, y / 224, z / 224
        ax.annotate("",
                    xy=(camera_lens[0] + 25, y_pos + 80),
                    xytext=(x_pos + 70, y_pos + 30),
                    arrowprops=dict(
                        arrowstyle="->,head_length=0.5,head_width=0.5", lw=3, color='b',
                        connectionstyle='bar,angle=180,fraction=-0.35'
                    ))
        ax.text(camera_lens[0] + 10, y_pos - 55, f"Next action: (x: {x:.2f}, y: {y:.2f}, scale: {z:.2f})",
                fontsize=20, color='b')

    x_pos += 148

    if stage >= FrameEnum.PREDICTION:
        ax.arrow(x_pos, y_pos + 112, 25, 0, length_includes_head=True, width=1, head_width=5)

    x_pos += 35

    if stage >= FrameEnum.PREDICTION:
        out = out[-1]
    else:
        if out.shape[0] == 2:
            out = out[0]
        else:
            out = None
    if out is not None:
        ax.imshow(out.permute((1, 2, 0)), extent=(x_pos, x_pos + 224, y_pos, y_pos + 224))
    ax.text(x_pos + 70, y_pos + 235, "Prediction", fontsize=16)


def animate_one(model: BaseRlMAE, image: Tensor, out: Tensor, coords: Tensor, patches: Tensor, scores: Tensor,
                target: Tensor, done: Tensor, save_path: str, rev_normalizer) -> None:
    num_glimpses = model.num_glimpses
    glimpse_grid_size = model.glimpse_grid_size
    patches_per_glimpse = coords.shape[0] // num_glimpses
    assert patches_per_glimpse * num_glimpses == coords.shape[0]  # assert if divisible by num_glimpses

    if len(out.shape) == 3:
        # reconstruction
        out = model.mae.unpatchify(out)
        out = rev_normalizer(out).to(torch.uint8)
    else:
        raise NotImplementedError()

    coords = [Coords.from_tensor(x) for x in coords]
    merged_coords = [Coords.merge(coords[idx * patches_per_glimpse: idx * patches_per_glimpse + patches_per_glimpse])
                     for idx in range(num_glimpses)]

    fig, ax = plt.subplots(figsize=(20, 7))
    fig.tight_layout()

    def update_fig(step_idx):
        ax.cla()
        if step_idx == 0:
            draw_frame(ax, FrameEnum.IMAGE, image, None, glimpse_grid_size, [], out[0:1],
                       merged_coords[0:1])
            return
        if step_idx < 3:
            draw_frame(ax, FrameEnum(len(FrameEnum) - 3 + step_idx), image, None, glimpse_grid_size, [], out[0:1],
                       merged_coords[0:1])
            return
        if step_idx >= 2 + num_glimpses * len(FrameEnum):
            draw_frame(ax, FrameEnum.PREDICTION, image, merged_coords[-1], glimpse_grid_size, merged_coords,
                       out[-2:], merged_coords[-1:])
            return

        glimpse_idx = (step_idx - 3) // len(FrameEnum)
        stage = FrameEnum((step_idx - 3) % len(FrameEnum))
        draw_frame(ax, stage, image, merged_coords[glimpse_idx], glimpse_grid_size, merged_coords[:glimpse_idx + 1],
                   out[glimpse_idx:glimpse_idx + 2], merged_coords[glimpse_idx:glimpse_idx + 2])

    max_steps = sum(not x for x in done)
    # noinspection PyTypeChecker
    anim = animation.FuncAnimation(fig, update_fig, max_steps * len(FrameEnum) + 8)
    anim.save(save_path, fps=1, savefig_kwargs={'pad_inches': 0})
    plt.close(fig)
