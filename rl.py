import datetime
import os
import platform
import time

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from architectures.rl_glimpse import RlMAE
from datasets import ImageNet1k

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    data = ImageNet1k(data_dir='/home/adam/datasets/imagenet', train_batch_size=32, eval_batch_size=32,
                      num_workers=8, always_drop_last=True)
    model = RlMAE(data, pretrained_mae_path='elastic_mae.ckpt', num_glimpses=12, rl_iters_per_step=1, batch_size=32,
                  epochs=10, init_random_batches=100, rl_batch_size=64,
                  replay_buffer_size=10000)

    plugins = []

    run_name = f'{time.strftime("%Y-%m-%d_%H:%M:%S")}-{platform.node()}'
    print('Run name:', run_name)

    loggers = []

    loggers.append(TensorBoardLogger(save_dir='logs/', name=run_name))

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

    callbacks = [
        ModelCheckpoint(dirpath=f"checkpoints/{run_name}", monitor='val/mae_rmse'),
        RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format='.2e')),
        RichModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = Trainer(plugins=plugins,
                      max_epochs=10,
                      accelerator='gpu',
                      logger=loggers,
                      callbacks=callbacks,
                      enable_model_summary=False,
                      strategy=strategy,
                      num_nodes=num_nodes,
                      devices=devices)

    trainer.fit(model=model)
