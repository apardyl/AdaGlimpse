# AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale
___
## Setup

```shell
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
* 
```shell
python train.py <dataset> <model> [params]
```
where dataset is one of:
* Reconstruction task:
  * ADE20KReconstruction
  * Coco2014Reconstruction
  * Sun360Reconstruction
  * TestImageDirReconstruction (any directory with img files)
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
