# AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale
___
## Setup
```shell
conda env create -f environment.yml -n AdaGlimpse # we recommend using mamba instead of conda (better performance)
conda activate AdaGlimpse
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
  * TestImageDirReconstruction
  * ImageNet1k
  * ElasticImageNet1k (use ImageNet1k files as data directory)
* Segmentation task:
  * ADE20KSegmentation
* Classification task:
  * Sun360Classification
  * ImageNet1k
  * ElasticImageNet1k (use ImageNet1k files as data directory)

and the model is:
* ClassificationRlMAE for classification
* ReconstructionRlMAE for reconstruction with RL as described in paper
* {DivideFour/StamLike}{Glimpse/SaliencyGlimpse}ElasticMae for reconstruction without RL
* SimpleAMEGlimpseElasticMae for reconstruction based on attention map entropy (AME)
* ElasticMae/HybridElasticMae for ???
* SegmentationRlMAE for segmentation

Example:
Run ReconstructionRlMAE on Elastic ImageNet1k with reconstruction task
```shell
python train.py ElasticImageNet1k  ReconstructionRlMAE --data-dir DATASET_DIR
```
---
Run `python train.py <dataset> <model> --help` for available training params.

Visualizations form the paper can be generated using `predict.py` 
(use `--help` param for more information and use ImageNet1k instead of ElasticImageNet1k here).
