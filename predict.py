import argparse
import os.path
import random
import sys

import torch
from lightning import Trainer
from tqdm import tqdm

from architectures.rl_glimpse import BaseRlMAE
from architectures.utils import RevNormalizer
from utils.prediction_hooks import RLStateReplaceHook, RLUserHook
from utils.prepare import experiment_from_args
from utils.visualisation_animate import animate_one
from utils.visualisation_grid import visualize_grid

random.seed(1)
torch.manual_seed(1)
torch.set_float32_matmul_precision("high")


def define_args(parent_parser):
    parser = parent_parser.add_argument_group("predict.py")
    parser.add_argument(
        "--visualization-path",
        help="path to save visualizations to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--animate",
        help="animate visualizations",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--model-checkpoint",
        help='path to a saved model state',
        type=str,
        required=True
    )
    parser.add_argument(
        "--random-samples",
        help='sample random images from the dataset',
        type=int,
        default=None
    )
    parser.add_argument(
        "--save-all",
        help='save all visualisation elements',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--dump-avg-state",
        help='dump average state values',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--replace-state",
        help='replace state element with value from file',
        type=str,
        default=None,
        choices=['patches', 'coords', 'importance', 'latent']
    )
    return parent_parser


def visualize(visualization_path, data_hook, model, save_all=False, animate=False):
    data = data_hook.compute()

    rev_normalizer = RevNormalizer()
    images = rev_normalizer(data["images"]).to(torch.uint8)
    patches = rev_normalizer(data["patches"]).to(torch.uint8)

    for idx, (img, out, coord, patch, score, target, done) in enumerate(tqdm(
            zip(images, data["out"], data["coords"], patches, data['scores'], data['targets'], data['done']),
            total=images.shape[0])):
        if animate:
            animate_one(model, img, out, coord, patch, score, target, done,
                        os.path.join(visualization_path, f"{idx}.mp4"), rev_normalizer)
        else:
            visualize_grid(model, img, out, coord, patch, score, target, done,
                           os.path.join(visualization_path, f"{idx}.png"), rev_normalizer, save_all)


def dump_avg_state(data_hook):
    data = data_hook.compute()
    avg_patch = data["patches"].mean(dim=0).mean(dim=0)
    avg_coords = data["coords"].mean(dim=0).mean(dim=0)
    avg_importance = data["importance"][-1].mean(dim=0).mean(dim=0)
    avg_latent = data["latent"].mean(dim=0).mean(dim=0)
    torch.save({
        "avg_patch": avg_patch,
        "avg_coords": avg_coords,
        "avg_importance": avg_importance,
        "avg_latent": avg_latent
    }, 'avg_state.pck')


def main():
    model: BaseRlMAE
    data_module, model, args = experiment_from_args(
        sys.argv, add_argparse_args_fn=define_args
    )

    data_module.num_random_eval_samples = args.random_samples
    model.parallel_games = 0

    model.load_pretrained(args.model_checkpoint)

    model.eval()

    if not isinstance(model, BaseRlMAE):
        raise RuntimeError(f"Unrecognized model type: {type(model)}")

    data_hook = None
    if args.visualization_path is not None or args.dump_avg_state:
        data_hook = model.add_user_forward_hook(RLUserHook(avg_latent=args.dump_avg_state))

    if args.replace_state is not None:
        avg_state = torch.load('avg_state.pck')
        replacement = {
            'patches': 'avg_patch',
            'coords': 'avg_coords',
            'importance': 'avg_importance',
            'latent': 'avg_latent'
        }[args.replace_state]
        model.add_user_forward_hook(RLStateReplaceHook(**{replacement: avg_state[replacement]}))

    trainer = Trainer()
    trainer.test(model)

    if args.dump_avg_state:
        dump_avg_state(data_hook)
        return

    if args.visualization_path is not None:
        visualization_path = args.visualization_path
        os.makedirs(visualization_path, exist_ok=True)
        visualize(visualization_path, data_hook, model, save_all=args.save_all, animate=args.animate)
        return


if __name__ == "__main__":
    main()
