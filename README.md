## Person_reID_baseline_pytorch

Baseline Code for Person-reID.

We arrived **Rank@1=88.24%** without bell and whistle.

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

## Dataset
[Market1501]()

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

## Ablation Study
Input is resized to 256x128

| BatchSize | Dropout | Rank@1 | mAP | Reference|
| --------- | -------- | ----- | ---- | ---- |
| 16 | 0.5  | 86.67 | 68.19 | |
| 32 | 0.5  | 87.98 | 69.38 | |
| 32 | 0.5  | 88.24 | 70.68 | test with 144x288|
| 32 | 0.5  | 87.14 | 68.90 | 0.1 color jitter|
| 64 | 0.5  | 86.82 | 67.48 | |
| 64 | 0.5  | 85.78 | 65.97 | 0.1 color jitter|
| 64 | 0.5  | 85.42 | 65.29 | 0.4 color jitter|
| 64 | 0.75 | 84.86 | 66.06 | |
| 96 | 0.5  | 86.05 | 67.03 | |
| 96 | 0.75 | 85.66 | 66.44 | |
