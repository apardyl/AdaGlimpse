from dataclasses import dataclass
from operator import itemgetter
from typing import List

import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from skimage.color import label2rgb
from torchvision.transforms.v2.functional import resize


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


class GridField:
    size_ratio = .3

    def __init__(self, data):
        self.data = data

    def render(self, axs: Axes):
        raise NotImplementedError()


class ImageGridField(GridField):
    size_ratio = 3.5

    def __init__(self, data):
        super().__init__(data)
        if len(self.data.shape) == 3:
            if self.data.shape[0] == 3:
                self.data = self.data.permute((1, 2, 0))
                self.data = self.data.numpy()

    def render(self, axs):
        axs.imshow(self.data, resample=False)
        axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def save(self, path):
        plt.imsave(path, self.data)


class SegmentationGridField(GridField):
    size_ratio = 3.5

    def __init__(self, data):
        super().__init__(data)

        if len(self.data.shape) == 3:
            self.data = self.data.argmax(dim=0)

        self.data = label2rgb(self.data.numpy(), bg_label=255, bg_color=(0, 0, 0))

    def render(self, axs):
        axs.imshow(self.data, resample=False)
        axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def save(self, path):
        plt.imsave(path, self.data)


class ScoreGridField(GridField):
    def render(self, axs):
        axs.set_axis_off()
        axs.text(0.15, 0.4, f'Score: {float(self.data):.3f}', font={'size': 18})


class ClsPredictionGridField(GridField):
    def render(self, axs):
        self.data = torch.nn.functional.softmax(self.data, dim=-1)
        pred = torch.argmax(self.data, dim=-1)
        score = self.data[pred]
        axs.set_axis_off()
        axs.text(0.15, 0.3, f'Pred: {int(pred)}\nProb: {float(score):.3f}', font={'size': 18})


class ClsTargetGridField(GridField):
    def render(self, axs):
        axs.set_axis_off()
        axs.text(0.15, 0.4, f'Label: {int(self.data)}', font={'size': 18})


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
    boxes_per_glimpse = (len(coords) // len(merged_coords))
    colors = ["black"] * (len(coords) - boxes_per_glimpse) + ["red"] * boxes_per_glimpse
    merged_colors = ["black"] * (len(merged_coords) - 1) + ["red"]
    img = torchvision.utils.draw_bounding_boxes(img, boxes, colors=colors, width=1)
    img = torchvision.utils.draw_bounding_boxes(img, merged_boxes, colors=merged_colors, width=3)
    return img


def selection_map(mask, patch_size):
    while len(mask.shape) > 2:
        mask = mask.squeeze(0)
    mask = mask.detach().cpu().numpy()
    mask = mask.repeat(patch_size[1], axis=1).repeat(patch_size[0], axis=0)
    return torch.from_numpy(mask).unsqueeze(0)
