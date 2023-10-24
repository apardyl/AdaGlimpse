import argparse
import os
import platform
import random
import signal
import sys
import time

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning_fabric.plugins.environments import SLURMEnvironment

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

    callbacks = [
        ModelCheckpoint(dirpath=f"checkpoints/{run_name}", monitor="val/loss"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=3)
    ]

    if 'SLURM_NTASKS' in os.environ:
        strategy = 'ddp'
        num_nodes = int(os.environ['SLURM_NNODES'])
        devices = int(os.environ['SLURM_NTASKS'])
        callbacks.append(
            SLURMEnvironment(requeue_signal=signal.SIGHUP)
        )
        print(f'Running on slurm, {num_nodes} nodes, {devices} gpus')
    else:
        strategy = 'auto'
        num_nodes = 1
        devices = 'auto'

    trainer = Trainer(plugins=plugins,
                      max_epochs=args.epochs,
                      accelerator='gpu',
                      logger=loggers,
                      callbacks=callbacks,
                      enable_model_summary=False,
                      strategy=strategy,
                      num_nodes=num_nodes,
                      devices=devices
                      )

    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.load_model_path)

    if data_module.has_test_data:
        trainer.test(ckpt_path='best', datamodule=data_module)


if __name__ == "__main__":
    main()
