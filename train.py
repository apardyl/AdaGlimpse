import argparse
import platform
import random
import sys
import time

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.cuda.amp import GradScaler

from utils.prepare import experiment_from_args

random.seed(1)
torch.manual_seed(1)


def define_args(parent_parser):
    parser = parent_parser.add_argument_group('train.py')
    parser.add_argument('--load-model-path',
                        help='load model from pth',
                        type=str,
                        default=None)
    parser.add_argument('--wandb',
                        help='log to wandb',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--tensorboard',
                        help='log to tensorboard',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--ddp',
                        help='use DDP acceleration strategy',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--name',
                        help='experiment name',
                        type=str,
                        default=None)
    return parent_parser


def main():
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args)

    plugins = []

    run_name = args.name
    if run_name is None:
        run_name = f'{time.strftime("%Y-%m-%d_%H:%M:%S")}-{platform.node()}'
    print('Run name:', run_name)

    loggers = []
    if args.tensorboard:
        loggers.append(TensorBoardLogger(save_dir='logs/', name=run_name))
    if args.wandb:
        loggers.append(WandbLogger(project='elastic_glimpse', entity="ideas_cv", name=run_name))

    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{run_name}", monitor="val/loss")

    trainer = Trainer(plugins=plugins, max_epochs=args.epochs, accelerator='auto', logger=loggers,
                      callbacks=[checkpoint_callback, RichProgressBar(leave=True), RichModelSummary(max_depth=3)],
                      enable_model_summary=False,
                      strategy=DDPStrategy(find_unused_parameters=False) if args.ddp else 'auto')

    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.load_model_path)

    if data_module.has_test_data:
        trainer.test(ckpt_path='best', datamodule=data_module)


if __name__ == "__main__":
    main()
