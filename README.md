# AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale (ECCV 2024)
### [Project Page](https://io.pardyl.com/AdaGlimpse/) | [Paper](https://arxiv.org/abs/2404.03482) | [Checkpoints](https://huggingface.co/apardyl/AdaGlimpse/tree/main)
Official PyTorch implementation of the paper: "AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale"

![](https://io.pardyl.com/AdaGlimpse/static/images/preview.gif)

> **AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale**<br>
> Adam Pardyl, Michał Wronka, Maciej Wołczyk, Kamil Adamczewski, Tomasz Trzciński, Bartosz Zieliński<br>
> [https://arxiv.org/abs/2404.03482](https://arxiv.org/abs/2404.03482)<br>
> 
> **Abstract:** Active Visual Exploration (AVE) is a task that involves dynamically selecting observations (glimpses), which is critical to facilitate comprehension and navigation within an environment. While modern AVE methods have demonstrated impressive performance, they are constrained to fixed-scale glimpses from rigid grids. In contrast, existing mobile platforms equipped with optical zoom capabilities can capture glimpses of arbitrary positions and scales. To address this gap between software and hardware capabilities, we introduce AdaGlimpse. It uses Soft Actor-Critic, a reinforcement learning algorithm tailored for exploration tasks, to select glimpses of arbitrary position and scale. This approach enables our model to rapidly establish a general awareness of the environment before zooming in for detailed analysis. Experimental results demonstrate that AdaGlimpse surpasses previous methods across various visual tasks while maintaining greater applicability in realistic AVE scenarios.
___
## Setup

```shell
git clone https://github.com/apardyl/AdaGlimpse.git && cd AdaGlimpse
conda env create -f environment.yml -n AdaGlimpse # we recommend using mamba instead of conda (better performance)
conda activate AdaGlimpse
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install timm lightning wandb rich scipy -c conda-forge
conda config --set pip_interop_enabled True
pip install torchrl
```

## Train
* download and extract the requested dataset
* run training with:

```shell
python train.py <dataset> <model> [params]
```

where dataset is one of:
* Reconstruction task:
  * ADE20KReconstruction
  * Coco2014Reconstruction
  * Sun360Reconstruction
  * TestImageDirReconstruction (any directory with jpg files)
  * ImageNet1k
* Segmentation task:
  * ADE20KSegmentation
* Classification task:
  * Sun360Classification
  * ImageNet1k

and the model is:
* ClassificationRlMAE for classification
* ReconstructionRlMAE for reconstruction
* SegmentationRlMAE for segmentation

Example:
Run ReconstructionRlMAE on Elastic ImageNet1k with reconstruction task
```shell
python train.py ImageNet1k  ReconstructionRlMAE --data-dir DATASET_DIR
```
---
Run `python train.py <dataset> <model> --help` for available training params.

Visualizations form the paper can be generated using `predict.py` 
(use `--help` param for more information).

## Trained models
Training checkpoints can be downloaded from Hugging Face: [https://huggingface.co/apardyl/AdaGlimpse](https://huggingface.co/apardyl/AdaGlimpse/tree/main)
