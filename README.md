<h1 align="center"> Person_reID_baseline_pytorch </h1>

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/layumi/Person_reID_baseline_pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/Person_reID_baseline_pytorch/context:python)
[![Build Status](https://travis-ci.org/layumi/Person_reID_baseline_pytorch.svg?branch=master)](https://travis-ci.org/layumi/Person_reID_baseline_pytorch)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/layumi/Person_reID_baseline_pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/Person_reID_baseline_pytorch/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A tiny, friendly, strong baseline code for Person-reID (based on [pytorch](https://pytorch.org)).

- **Strong.** It is consistent with the new baseline result in several top-conference works, e.g., [Joint Discriminative and Generative Learning for Person Re-identification(CVPR19)](https://arxiv.org/abs/1904.07223), [Beyond Part Models: Person Retrieval with Refined Part Pooling(ECCV18)](https://arxiv.org/abs/1711.09349), [Camera Style Adaptation for Person Re-identification(CVPR18)](https://arxiv.org/abs/1711.10295). We arrived Rank@1=88.24%, mAP=70.68% only with softmax loss. 

- **Small.** With fp16 (supported by Nvidia apex), our baseline could be trained with only 2GB GPU memory.

- **Friendly.** You may use the off-the-shelf options to apply many state-of-the-art tricks in one line.
Besides, if you are new to person re-ID, you may check out our **[Tutorial](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial)** first (8 min read) :+1: .
![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/show.png)

## Table of contents
* [Features](#features)
* [Some News](#some-news)
* [Trained Model](#trained-model)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset--preparation)
    * [Train](#train)
    * [Test](#test)
    * [Evaluation](#evaluation)
* [Tips for training with other datasets](#tips)
* [Citation](#citation)
* [Related Repos](#related-repos)

## Features
Now we have supported:
- Float16 to save GPU memory based on [apex](https://github.com/NVIDIA/apex)
- Part-based Convolutional Baseline(PCB)
- Multiple Query Evaluation
- Re-Ranking
- Random Erasing
- ResNet/DenseNet
- Visualize Training Curves
- Visualize Ranking Result
- [Visualize Heatmap](https://github.com/layumi/Person_reID_baseline_pytorch/blob/dev/visual_heatmap.py)
- Linear Warm-up 

Here we provide hyperparameters and architectures, that were used to generate the result. 
Some of them (i.e. learning rate) are far from optimal. Do not hesitate to change them and see the effect. 

P.S. With similar structure, we arrived **Rank@1=87.74% mAP=69.46%** with [Matconvnet](http://www.vlfeat.org/matconvnet/). (batchsize=8, dropout=0.75) 
You may refer to [Here](https://github.com/layumi/Person_reID_baseline_matconvnet).
Different framework need to be tuned in a different way.

## Some News
**11 June 2020** People live in the 3D world. We release one new person re-id code [Person Re-identification in the 3D Space](https://github.com/layumi/person-reid-3d), which conduct representation learning in the 3D space. You are welcomed to check out it.

**30 April 2020** We have applied this code to the [AICity Challenge 2020](https://www.aicitychallenge.org/),  yielding the 1st Place Submission to the re-id track :red_car:. Check out [here](https://github.com/layumi/AICIty-reID-2020).

**01 March 2020** We release one new image retrieval dataset, called [University-1652](https://github.com/layumi/University1652-Baseline), for drone-view target localization and drone navigation :helicopter:. It has a similar setting with the person re-ID. You are welcomed to check out it.

**07 July 2019:** I added some new functions, such as `--resume`, auto-augmentation policy, acos loss, into [developing thread](https://github.com/layumi/Person_reID_baseline_pytorch/tree/dev) and rewrite the `save` and `load` functions. I haven't tested the functions throughly. Some new functions are worthy of having a try. If you are first to this repo, I suggest you stay with the master thread.

**01 July 2019:** [My CVPR19 Paper](https://arxiv.org/abs/1904.07223) is online. It is based on this baseline repo as teacher model to provide pseudo label for the generated images to train a better student model. You are welcomed to check out the opensource code at [here](https://github.com/NVlabs/DG-Net).

**03 Jun 2019:** Testing with multiple-scale inputs is added. You can use `--ms 1,0.9` when extracting the feature. It could slightly improve the final result.

**20 May 2019:** Linear Warm Up is added. You also can set warm-up the first K epoch by `--warm_epoch K`. If K <=0, there will be no warm-up.

**What's new:** FP16 has been added. It can be used by simply added `--fp16`. You need to install [apex](https://github.com/NVIDIA/apex) and update your pytorch to 1.0. 

Float16 could save about 50% GPU memory usage without accuracy drop. **Our baseline could be trained with only 2GB GPU memory.** 
```bash
python train.py --fp16
```
**What's new:** Visualizing ranking result is added.
```bash
python prepare.py
python train.py
python test.py
python demo.py --query_index 777
```

**What's new:** Multiple-query Evaluation is added. The multiple-query result is about **Rank@1=91.95% mAP=78.06%**. 
```bash
python prepare.py
python train.py
python test.py --multi
python evaluate_gpu.py
```

**What's new:**  [PCB](https://arxiv.org/abs/1711.09349) is added. You may use '--PCB' to use this model. It can achieve around **Rank@1=92.73% mAP=78.16%**. I used a GPU (P40) with 24GB Memory. You may try apply smaller batchsize and choose the smaller learning rate (for stability) to run. (For example, `--batchsize 32 --lr 0.01 --PCB`)
```bash
python train.py --PCB --batchsize 64 --name PCB-64
python test.py --PCB --name PCB-64
```

**What's new:** You may try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

**What's new:** You may apply '--use_dense' to use `DenseNet-121`. It can arrive around Rank@1=89.91% mAP=73.58%. 

**What's new:** Re-ranking is added to evaluation. The re-ranked result is about **Rank@1=90.20% mAP=84.76%**.

**What's new:** Random Erasing is added to train.

**What's new:** I add some code to generate training curves. The figure will be saved into the model folder when training.

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/train.jpg)

## Trained Model
I re-trained several models, and the results may be different with the original one. Just for a quick reference, you may directly use these models. 
The download link is [Here](https://drive.google.com/open?id=1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE).

|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| [ResNet-50] | 88.84% | 71.59% |  `python train.py --train_all` |
| [DenseNet-121] | 90.17% | 74.02% | `python train.py --name ft_net_dense --use_dense --train_all` |
| [PCB] | 92.64% | 77.47% | `python train.py --name PCB --PCB --train_all --lr 0.02` |
| [ResNet-50 (fp16)] | 88.03% | 71.40% | `python train.py --name fp16 --fp16 --train_all` |
| [ResNet-50 (all tricks)] | 91.83% | 78.32% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5` |

### Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- [Optional] apex (for float16) 
- [Optional] [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)

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
- [Optinal] You may skip it. Install apex from the source
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .

### Dataset & Preparation
Download [Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset]( [GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) or ([BaiduYun](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) password: bhbh)).
Our baseline code is not such high on DukeMTMC-reID **Rank@1=64.23%, mAP=43.92%**. Hyperparameters are need to be tuned.

### Train
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

### Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


### Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

### re-ranking
```bash
python evaluate_rerank.py
```
**It may take more than 10G Memory to run.** So run it on a powerful machine if possible. 

It will output Rank@1, Rank@5, Rank@10 and mAP results.

### Tips
Notes the format of the camera id and the number of cameras.

For some dataset, e.g., MSMT17, there are more than 10 cameras. You need to modify the `prepare.py` and `test.py` to read the double-digit camera ID.

For some vehicle re-ID datasets. e.g. VeRi, you also need to modify the `prepare.py` and `test.py`.  It has different naming rules.
https://github.com/layumi/Person_reID_baseline_pytorch/issues/107 (Sorry. It is in Chinese)


## Citation
The following paper uses and reports the result of the baseline model. You may cite it in your paper.
```
@article{zheng2019joint,
  title={Joint discriminative and generative learning for person re-identification},
  author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, Liang and Yang, Yi and Kautz, Jan},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

The following papers may be the first two to use the bottleneck baseline. You may cite them in your paper.
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

Basic Model
```
@article{zheng2018discriminatively,
  title={A discriminatively learned CNN embedding for person reidentification},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  volume={14},
  number={1},
  pages={13},
  year={2018},
  publisher={ACM}
}
```

## Related Repos
1. [Pedestrian Alignment Network](https://github.com/layumi/Pedestrian_Alignment) ![GitHub stars](https://img.shields.io/github/stars/layumi/Pedestrian_Alignment.svg?style=flat&label=Star)
2. [2stream Person re-ID](https://github.com/layumi/2016_person_re-ID) ![GitHub stars](https://img.shields.io/github/stars/layumi/2016_person_re-ID.svg?style=flat&label=Star)
3. [Pedestrian GAN](https://github.com/layumi/Person-reID_GAN) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person-reID_GAN.svg?style=flat&label=Star)
4. [Language Person Search](https://github.com/layumi/Image-Text-Embedding) ![GitHub stars](https://img.shields.io/github/stars/layumi/Image-Text-Embedding.svg?style=flat&label=Star)
5. [DG-Net](https://github.com/NVlabs/DG-Net) ![GitHub stars](https://img.shields.io/github/stars/NVlabs/DG-Net.svg?style=flat&label=Star)
6. [3D Person re-ID](https://github.com/layumi/person-reid-3d) ![GitHub stars](https://img.shields.io/github/stars/layumi/person-reid-3d.svg?style=flat&label=Star)
