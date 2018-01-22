## Person_reID_baseline_pytorch

Baseline Code (with bottleneck) for Person-reID (pytorch).
It is consistent with the new baseline result in [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349) and [Camera Style Adaptation for Person Re-identification](https://arxiv.org/abs/1711.10295).

We arrived **Rank@1=88.24%, mAP=70.68%** only with softmax loss.

Here we provide hyperparameters and architectures, that were used to generate the result. 
Some of them (i.e. learning rate) are far from optimal. Do not hesitate to change them and see the effect. 

P.S. With similar structure, we arrived **Rank@1=86.85% mAP=67.29%** with Matconvnet. (batchsize=8, dropout=0.75)
Different framework need to be tuned in a different way.

**What's new：** Trained model is available at [GoogleDrive](https://drive.google.com/open?id=1__x0qNJ3T654wTghmuRjydn42NsAZW_M).

**What's new:** Re-ranking is added to evaluation. The re-ranked result is **Rank@1=90.20% mAP=84.76%**.

**What's new:** Random Erasing is added to train.

**What's new:** I add some code to generate training curves. The figure will be saved into the model folder when training.

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/train.jpg)

## Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy

**(Some reports found that updating numpy can arrive the right accuracy. If you only get 50~80 Top1 Accuracy, just try it.)**
We have successfully run the code based on numpy 1.12.1 and 1.13.1 .

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
Download [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html)

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation).
Our baseline code is not such high on DukeMTMC-reID **Rank@1=64.23%, mAP=43.92%**. Hyperparameters are need to be tuned.

To save trained model, we make a dir.
```bash
mkdir model 
```

## Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

## Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


## Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

### re-ranking
```bash
python evaluate_rerank.py
```
**It may take more than 10G Memory to run.** So run it on a powerful machine if possible. 

It will output Rank@1, Rank@5, Rank@10 and mAP results.

## Ablation Study
The model is based on Resnet50. Input images are resized to 256x128.
Here we just show some results.

**Note that the result may contain around 1% bias.(For example, 50th-epoch model can be better.)**

| BatchSize | Dropout | Rank@1 | mAP | Note|
| --------- | -------- | ----- | ---- | ---- |
| 16 | 0.5  | 86.67 | 68.19 | |
| 32 | 0.5  | 87.98 | 69.38 | |
| 32 | 0.5  | **88.24** | **70.68** | test with 288x144|
| 32 | 0.5  | **89.13** | **73.50** | train with random erasing and test with 288x144|
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


## Citation
As far as I know, the following papers may be the first two to use the bottleneck baseline. You may cite them in your paper.
```
@article{DBLP:journals/corr/SunZDW17,
  author    = {Yifan Sun and
               Liang Zheng and
               Weijian Deng and
               Shengjin Wang},
  title     = {SVDNet for Pedestrian Retrieval},
  booktitle   = {ICCV},
  year      = {2017},
}

@article{hermans2017defense,
  title={In Defense of the Triplet Loss for Person Re-Identification},
  author={Hermans, Alexander and Beyer, Lucas and Leibe, Bastian},
  journal={arXiv preprint arXiv:1703.07737},
  year={2017}
}
```
