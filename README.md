## Person_reID_baseline_pytorch

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/bottleneck.png)
New Baseline Code (with bottleneck) for Person-reID (pytorch).
It is consistent with the baseline result in [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349).

We arrived **Rank@1=88.24%, mAP=70.68%** without bell and whistle.

[[MatConvnet Version]]() 
We also arrived **Rank@1=86.85%, mAP=67.29%**. The code will come soon.

Here we provide hyperparameters and architectures, that were used to generate the result. 
Some of them are far from optimal. Do not hesitate to change them and see the effect.

## Prerequisites

- Python 3.6
- CUDA 8.0
- GPU Memory >= 6G

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0 and Torchvision 0.2.0.

## Dataset & Preparation
[Market1501](http://www.liangzheng.org/Project/project_reid.html)
Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remeber to change the dataset path to your own path.

## Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 64  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

## Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --which_epoch 50
```
`--gpu_ids` which gpu to run.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


## Evaluation
```bash
python evaluation.py
```

## Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Ablation Study
Input is resized to 256x128

| BatchSize | Dropout | Rank@1 | mAP | Note|
| --------- | -------- | ----- | ---- | ---- |
| 16 | 0.5  | 86.67 | 68.19 | |
| 32 | 0.5  | 87.98 | 69.38 | |
| 32 | 0.5  | **88.24** | **70.68** | test with 144x288|
| 32 | 0.5  | 87.14 | 68.90 | 0.1 color jitter|
| 64 | 0.5  | 86.82 | 67.48 | |
| 64 | 0.5  | 85.78 | 65.97 | 0.1 color jitter|
| 64 | 0.5  | 85.42 | 65.29 | 0.4 color jitter|
| 64 | 0.75 | 84.86 | 66.06 | |
| 96 | 0.5  | 86.05 | 67.03 | |
| 96 | 0.75 | 85.66 | 66.44 | |

### Bottleneck
Test with 144x288, dropout rate is 0.5

| BatchSize | Bottleneck | Rank@1 | mAP | Note|
| --------- | ---------- | ------ | --- | ---- |
| 32 | 256  | 87.26 | 69.92 | |
| 32 | 512  | **88.24** | **70.68** | |
| 32 | 1024 | 84.29 | 64.00 | |
