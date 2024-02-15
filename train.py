import argparse
import datetime
import os
import platform
import sys
import time

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from architectures.base import AutoconfigLightningModule
from utils.prepare import experiment_from_args

torch.set_float32_matmul_precision('high')


def define_args(parent_parser):
    parser = parent_parser.add_argument_group('train.py')
    parser.add_argument('--wandb',
                        help='log to wandb (else use tensorboard)',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--fp16',
                        help='use 16 bit precision',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--name',
                        help='experiment name',
                        type=str,
                        default=None)
    parser.add_argument('--validate-only',
                        help='perform only the validation step',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--test-only',
                        help='perform only the validation step',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--resume',
                        help='resume from checkpoint',
                        type=str,
                        default=None)
    return parent_parser


def main():
    model: AutoconfigLightningModule
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args)

    plugins = []

    run_name = args.name
    if run_name is None:
        run_name = f'{time.strftime("%Y-%m-%d_%H:%M:%S")}-{platform.node()}'
    print('Run name:', run_name)

    loggers = []

    if args.wandb:
        loggers.append(WandbLogger(project='elastic_glimpse', entity="ideas_cv", name=run_name))
    else:
        loggers.append(TensorBoardLogger(save_dir='logs/', name=run_name))

    callbacks = [
        ModelCheckpoint(dirpath=f"checkpoints/{run_name}", monitor=model.checkpoint_metric,
                        mode=model.checkpoint_metric_mode, save_last=True, save_top_k=1, every_n_epochs=1),
        RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format='.2e')),
        RichModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval='step')
    ]

    if 'SLURM_NTASKS' in os.environ:
        num_nodes = int(os.environ['SLURM_NNODES'])
        devices = int(os.environ['SLURM_NTASKS'])
        if num_nodes * devices > 1:
            strategy = DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=3600))
        else:
            strategy = 'auto'
        print(f'Running on slurm, {num_nodes} nodes, {devices} gpus')
    else:
        strategy = 'auto'
        num_nodes = 1
        devices = 'auto'

    if not args.fp16:
        precision = None
    elif torch.cuda.is_bf16_supported():
        precision = 'bf16-mixed'
    else:
        precision = '16-mixed'

    trainer = Trainer(plugins=plugins,
                      max_epochs=args.epochs,
                      accelerator='gpu',
                      logger=loggers,
                      callbacks=callbacks,
                      enable_model_summary=False,
                      strategy=strategy,
                      num_nodes=num_nodes,
                      devices=devices,
                      precision=precision,
                      use_distributed_sampler=not model.internal_data)

    if not model.internal_data:
        kwargs = {
            'model': model,
            'datamodule': data_module
        }
    else:
        kwargs = {
            'model': model
        }

    if args.validate_only:
        trainer.validate(**kwargs, ckpt_path=args.resume)
        return

    if args.test_only:
        trainer.test(**kwargs, ckpt_path=args.resume)
        return

    trainer.fit(**kwargs, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
