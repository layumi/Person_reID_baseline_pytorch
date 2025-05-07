<h1 align="center"> Pytorch ReID </h1>
<h2 align="center"> Strong, Small, Friendly </h2>

![Python3.6+](https://img.shields.io/badge/python-3.6+-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A tiny, friendly, strong baseline code for Object-reID (based on [pytorch](https://pytorch.org)) since 2017.

- **Strong.** It is consistent with the new baseline result in several top-conference works, e.g., [Joint Discriminative and Generative Learning for Person Re-identification(CVPR19)](https://arxiv.org/abs/1904.07223), [Beyond Part Models: Person Retrieval with Refined Part Pooling(ECCV18)](https://arxiv.org/abs/1711.09349), [Camera Style Adaptation for Person Re-identification(CVPR18)](https://arxiv.org/abs/1711.10295). We arrived Rank@1=88.24%, mAP=70.68% only with softmax loss. 

- **Small.** With bf16/fp16 (supported by native pytorch), our baseline could be trained with only 2GB GPU memory.

- **Friendly.** You may use the off-the-shelf options to apply many state-of-the-art tricks in one line.
Besides, if you are new to object re-ID, you may check out our **[Tutorial](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial)** first (8 min read) :+1: .
![Person ReID Demo](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/show.png)

Share to
    <!-- Facebook -->
    <a href="http://www.facebook.com/sharer.php?u=https://github.com/layumi/Person_reID_baseline_pytorch/" target="_blank">
        <img src="https://github.com/layumi/Person_reID_baseline_pytorch/raw/master/docs/1_Facebook_colored_svg_copy-64.webp" alt="Facebook" width="60"/>
    </a>
    <!-- Twitter -->
    <a href="https://twitter.com/share?url=https://github.com/layumi/Person_reID_baseline_pytorch/&amp;text=Strong,%20Small,%20Friendly%20Pytorch%20re-ID&amp;hashtags=pytorch re-ID" target="_blank">
        <img src="https://github.com/layumi/Person_reID_baseline_pytorch/raw/master/docs/2018_social_media_popular_app_logo_twitter-64.webp" alt="Twitter" width="60"/>
    </a>
    <!-- Weibo -->
    <a href="https://service.weibo.com/share/share.php?url=https://github.com/layumi/Person_reID_baseline_pytorch/&amp;text=Strong,%20Small,%20Friendly%20Pytorch%20re-ID&amp;hashtags=pytorch re-ID" target="_blank">
        <img src="https://github.com/layumi/Person_reID_baseline_pytorch/raw/master/docs/Weibo-64.webp" alt="Weibo" width="60" />
    </a>
    <!-- LinkedIn -->
    <a href="http://www.linkedin.com/shareArticle?mini=true&amp;url=https://github.com/layumi/Person_reID_baseline_pytorch/" target="_blank">
        <img src="https://github.com/layumi/Person_reID_baseline_pytorch/raw/master/docs/circle-linkedin-64.webp" alt="LinkedIn" width="60" />
    </a>
    <!-- Email -->
    <a href="mailto:?Subject=Strong, Small, Friendly Pytorch Re-identification&amp;Body=I%20saw%20this%20and%20thought%20of%20you!%20 https://github.com/layumi/Person_reID_baseline_pytorch/">
        <img src="https://github.com/layumi/Person_reID_baseline_pytorch/raw/master/docs/Email-64.webp" alt="Email" width="60"/>
    </a>
        

## Tutorial
* [8 min Tutorial](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/tutorial/README.md)，[8分钟教程](https://zhuanlan.zhihu.com/p/50387521)
* [中文视频简介](https://www.bilibili.com/video/BV11K4y1f7eQ)
* [Answer to Tutorial](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/tutorial/Answers_to_Quick_Questions.md)

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
* [How to Cite?](#citation)
* [Related Repos](#related-repos)

## Features
Now we have supported:

### Training 
- bf16 and fp16 (Float16) to save GPU memory based on native pytorch (replace apex).
- Running the code on Google Colab with Free GPU. Check [Here](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/colab) (Thanks to @ronghao233)
- [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) (10x Large Synthetic Dataset from Market **CVPR 2019 Oral**)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) / [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) / [HRNet](https://github.com/HRNet)
- ResNet/ResNet-ibn/DenseNet
- Circle Loss, Triplet Loss, Contrastive Loss, Sphere Loss, Lifted Loss, Arcface, Cosface  and Instance Loss
- Part-based Convolutional Baseline(PCB)
- Random Erasing
- Linear Warm-up 
- torch.compile (faster training)
- DDP (Multiple GPUs)

### Testing
- TensorRT 
- Pytorch JIT 
- Fuse Conv and BN layer into one Conv layer
- Multiple Query Evaluation
- Re-Ranking (CPU & [GPU Version](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/GPU-Re-Ranking))
- Visualize Training Curves
- Visualize Ranking Result
- [Visualize Heatmap](https://github.com/layumi/Person_reID_baseline_pytorch/blob/dev/visual_heatmap.py)


Here we provide hyperparameters and architectures, that were used to generate the result. 
Some of them (i.e. learning rate) are far from optimal. Do not hesitate to change them and see the effect. 

P.S. With similar structure, we arrived **Rank@1=87.74% mAP=69.46%** with [Matconvnet](http://www.vlfeat.org/matconvnet/). (batchsize=8, dropout=0.75) 
You may refer to [Here](https://github.com/layumi/Person_reID_baseline_matconvnet).
Different framework need to be tuned in a slightly different way.

## Some News


**22 Dec 2024** We are holding a workshop at ACM WWW 2025 on Multimedia Object Re-ID. You are welcome to show your insights. https://www.zdzheng.xyz/MORE2025/ 
Submission DDL is **1 Jan 2025**.

**12 Jan 2024** We are holding a workshop at ACM ICMR 2024 on  Multimedia Object Re-ID. You are welcome to show your insights. See you at Phuket, Thailand!😃 The workshop link is https://www.zdzheng.xyz/MORE2024/ . Submission DDL is **15 April 2024**. Good papers will be recommended to [ACM TOMM Special Issue](https://dl.acm.org/pb-assets/static_journal_pages/tomm/pdf/ACM-SI_ToMM_MMGR-1708635711467.pdf ) (CCF-B). (Re-submission is needed.)

<details>
 <summary><b>
  2023 News
</b></summary>
    
**12 Aug 2023** Large Person Langauge Model is currently available at [Here](https://github.com/Shuyu-XJTU/APTM)![GitHub stars](https://img.shields.io/github/stars/Shuyu-XJTU/APTM.svg?style=flat&label=Star) accepted by ACM MM'23. You are welcomed to check it.

**19 Mar 2023** We host a special session on IEEE Intelligent Transportation Systems Conference (ITSC), covering the object re-identification & point cloud topic. The paper ddl is by May 15, 2023 and the paper notification is at June 30, 2023. Please select the session code ``w7r4a'' during submission. More details can be found at [Special Session Website](https://2023.ieee-itsc.org/wp-content/uploads/2023/03/IEEE-ITSC-2023-Special-Session-Proposal-Safe-Critical-Scenario-Understanding-in-Intelligent-Transportation-Systems-SCSU-ITS.pdf).  

**9 Mar 2023** Market-1501 is in 3D. Please check our single 2D to 3D reconstruction work  https://github.com/layumi/3D-Magic-Mirror ![GitHub stars](https://img.shields.io/github/stars/layumi/3D-Magic-Mirror.svg?style=flat&label=Star).

![Magic Mirror](https://zdzheng.xyz/resource-img/rainbow_github.webp)

</details>

<details>
 <summary><b>
  2022 News
</b></summary>

**7 Sep 2022** We support SwinV2. 

**24 Jul 2022** Market-HQ is released with super-resolution quality from 128\*64 to 512\*256. Please check at https://github.com/layumi/HQ-Market

**14 Jul 2022** Add adversarial training by ``python train.py --name ftnet_adv --adv 0.1 --aiter 40``. 

**1 Feb 2022** Speed up the inference process about 10 seconds by removing the ``cat`` function in ``test.py``. 
   
**1 Feb 2022** Add the demo with ``TensorRT`` (The fast inference speed may depend on the GPU with the latest RT Core).
   
</details>   

<details>
 <summary><b>
  2021 News
 </b></summary>

**30 Dec 2021** We add supports for new losses, including arcface loss, cosface loss and instance loss. The hyper-parameters are still tunning.
   
**3 Dec 2021** We add supports for four losses, including triplet loss, contrastive loss, sphere loss and lifted loss. The hyper-parameters are still tunning.
   
**1 Dec 2021** We support EfficientNet/HRNet.
   
**15 Sep 2021** We support ResNet-ibn from ECCV2018 (https://github.com/XingangPan/IBN-Net). 

**17 Aug 2021** We support running code on Google Colab with free GPU. Please check it out at https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/colab .

**14 Aug 2021** We have supported the training with [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) for regularization via [Self-supervised Memory Learning](https://www.ijcai.org/proceedings/2020/150). You only neeed to download/unzip the dataset and add `--DG` to train model. 

**12 Aug 2021** We have supported the transformer-based model `Swin` by `--use_swin`. The basic performance is 92.73% Rank@1 and 79.71%mAP.

**23 Jun 2021** Attack your re-ID model via Query! They are not robust as you expected! Check the code at [Here](https://github.com/layumi/A_reID).

**5 Feb 2021** We have supported [Circle loss](https://arxiv.org/abs/2002.10857)(CVPR20 Oral). You can try it by simply adding `--circle`.  

**11 January 2021** On the Market-1501 dataset, we accelerate the re-ranking processing from **89.2s** to **9.4ms** with one K40m GPU, facilitating the real-time post-processing. The pytorch implementation can be found in [GPU-Re-Ranking](GPU-Re-Ranking/).

</details>

<details>
 <summary><b>
  2020 News
 </b></summary>
   
**11 June 2020** People live in the 3D world. We release one new person re-id code [Person Re-identification in the 3D Space](https://github.com/layumi/person-reid-3d), which conduct representation learning in the 3D space. You are welcomed to check out it.

<img width="250" height="150" src="https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/pdf/3D-demo.png"/>


**30 April 2020** We have applied this code to the [AICity Challenge 2020](https://www.aicitychallenge.org/),  yielding the 1st Place Submission to the re-id track :red_car:. Check out [here](https://github.com/layumi/AICIty-reID-2020).

**01 March 2020** We release one new image retrieval dataset, called [University-1652](https://github.com/layumi/University1652-Baseline), for drone-view target localization and drone navigation :helicopter:. It has a similar setting with the person re-ID. You are welcomed to check out it.
</details>

<details>
 <summary><b>
  2019 News
 </b></summary>
   
**07 July 2019:** I added some new functions, such as `--resume`, auto-augmentation policy, acos loss, into [developing thread](https://github.com/layumi/Person_reID_baseline_pytorch/tree/dev) and rewrite the `save` and `load` functions. I haven't tested the functions throughly. Some new functions are worthy of having a try. If you are first to this repo, I suggest you stay with the master thread.

**01 July 2019:** [My CVPR19 Paper](https://arxiv.org/abs/1904.07223) is online. It is based on this baseline repo as teacher model to provide pseudo label for the generated images to train a better student model. You are welcomed to check out the opensource code at [here](https://github.com/NVlabs/DG-Net).

**03 Jun 2019:** Testing with multiple-scale inputs is added. You can use `--ms 1,0.9` when extracting the feature. It could slightly improve the final result.

**20 May 2019:** Linear Warm Up is added. You also can set warm-up the first K epoch by `--warm_epoch K`. If K <=0, there will be no warm-up.

 </details>
   
<details>
 <summary><b>
  2018 & 2017 News
 </b></summary> 
   
**What's new:** FP16 has been added. It can be used by simply added `--fp16`. You need to update your pytorch to 2.0. 

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
 </details>
   
## Trained Model
I re-trained several models, and the results may be different with the original one. Just for a quick reference, you may directly use these models. 
The download link is [Here](https://drive.google.com/open?id=1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE).

|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| [EfficientNet-b4] | 85.78% | 66.80% |  `python train.py --use_efficient --name eff; python test.py --name eff` |
| [ResNet-50 + adv defense] | 87.77% | 69.83% |  `python train.py  --name adv0.1_40_w10_all --adv 0.1 --aiter 40 --warm 10 --train_all; python test.py  --name adv0.1_40_w10_all` |
| [ConvNeXt] | 88.98% | 71.35% |  `python train.py --use_convnext --name convnext; python test.py --name convnext` |
| [ResNet-50 (fp16)] | 88.03% | 71.40% | `python train.py --name fp16 --fp16 --train_all` |
| [ResNet-50] | 88.84% | 71.59% |  `python train.py --train_all` |
| [ResNet-50 + USAM] | 88.75% | 71.95% |  `python train.py --train_all --name res-usam --usam; python test.py --name res-usam` |
| [ResNet-50-ibn] | 89.13% | 73.40% | `python train.py --train_all --name res-ibn --ibn` |
| [ResNet-50-ibn + USAM] | 89.01% | 73.54% | `python train.py --train_all --name ibn-usam --ibn --usam; python test.py --name ibn-usam` |
| [DenseNet-121] | 90.17% | 74.02% | `python train.py --name ft_net_dense --use_dense --train_all` |
| [DenseNet-121 (Circle)] | 91.00% | 76.54% | `python train.py --name ft_net_dense_circle_w5 --circle --use_dense --train_all --warm_epoch 5` |
| [HRNet-18] | 90.83% | 76.65% |  `python train.py --use_hr --name hr18; python test.py --name hr18` |
| [PCB] | 92.64% | 77.47% | `python train.py --name PCB --PCB --train_all --lr 0.02` |
| [PCB + DG] | 92.70% | 78.31% | `python train.py --name PCB_DG --PCB --train_all --lr 0.02 --DG; python test.py --name PCB_DG` |
| [ResNet-50 (all tricks)] | 91.83% | 78.32% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5` |
| [ResNet-50 (all tricks+Circle)] | 92.13% | 79.84% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5_circle  --circle` |
| [ResNet-50 (all tricks+Circle+DG)] | 92.13% | 80.13% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5_circle_DG --circle --DG; python test.py --name warm5_s1_b8_lr2_p0.5_circle_DG` |
| [DenseNet-121 (all tricks+Circle)] | 92.61% | 80.24% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name dense_warm5_s1_b8_lr2_p0.5_circle --circle --use_dense; python test.py --name  dense_warm5_s1_b8_lr2_p0.5_circle` |
| [HRNet-18 (all tricks+Circle+DG)]| 92.19% | 81.00% | `python train.py --use_hr --name  hr18_p0.5_circle_w5_b16_lr0.01_DG --lr 0.01 --batch 16 --DG --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name  hr18_p0.5_circle_w5_b16_lr0.01_DG` |
| [Swin] (224x224) | 92.75% | 79.70% | `python train.py --use_swin --name swin; python test.py --name swin`|
| [SwinV2 (all tricks+Circle 256x128)] | 92.93% | 82.99% | `python train.py --use_swinv2 --name swinv2_p0.5_circle_w5_b16_lr0.03  --lr 0.03 --batch 16 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name   swinv2_p0.5_circle_w5_b16_lr0.03 --batch 32`|
| [Swin (all tricks+Circle 224x224)] | 94.12% | 84.39% | `python train.py --use_swin --name swin_p0.5_circle_w5  --erasing_p 0.5 --circle --warm_epoch 5;  python test.py --name swin_p0.5_circle_w5`|
| [Swin (all tricks+Circle+b16 224x224)] | 94.00% | 85.21% | `python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01 --lr 0.01 --batch 16  --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01`|
| [Swin (all tricks+Circle+b16+DG 224x224)] | 94.00% | 85.36% | `python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01_DG --lr 0.01 --batch 16 --DG --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01_DG`|

* More training iterations may lead to better results. 
* Swin costs more GPU memory (11G GPU is needed) to run. 
* The hyper-parameter of [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) `--DG` is not tuned. Better hyper-parameter may lead to better results.

### Different Losses 
   
I do not optimize the hyper-parameters. You are free to tune them for better performance.

|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| CE | 92.01% | 79.31% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_100 --total 100 ; python test.py  --name  warm5_s1_b32_lr8_p0.5_100`|
| CE + Sphere [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf) | 92.01% | 79.39% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_sphere100 --sphere --total 100; python test.py --name warm5_s1_b32_lr8_p0.5_sphere100` |
| CE + Triplet [[Paper]](https://arxiv.org/pdf/1703.07737) | 92.40%	| 79.71% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_triplet100 --triplet --total 100; python test.py  --name warm5_s1_b32_lr8_p0.5_triplet100` |
| CE + Lifted [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)|  91.78% | 79.77% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_lifted100 --lifted --total 100; python test.py --name warm5_s1_b32_lr8_p0.5_lifted100` |
| CE + Instance [[Paper]](https://zdzheng.xyz/files/TOMM20.pdf) | 92.73% | 81.11% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_instance100_gamma64 --instance --ins_gamma 64 --total 100 ; python test.py  --name  warm5_s1_b32_lr8_p0.5_instance100_gamma64`|
| CE + Contrast [[Paper]](https://zdzheng.xyz/files/TOMM18.pdf) | 92.28% | 81.42% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_contrast100 --contrast  --total 100; python test.py  --name warm5_s1_b32_lr8_p0.5_contrast100`|
| CE + Circle [[Paper]](https://arxiv.org/abs/2002.10857) | 92.46% | 81.70% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_circle100 --circle --total 100 ; python test.py  --name  warm5_s1_b32_lr8_p0.5_circle100` |
| CE + Contrast + Sphere | 92.79% | 82.02% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_cs100 --contrast --sphere --total 100; python test.py --name warm5_s1_b32_lr8_p0.5_cs100`|
| CE + Contrast + Triplet (Long) | 92.61% | 82.01% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 24 --lr 0.062 --name warm5_s1_b24_lr6.2_p0.5_contrast_triplet_133 --contrast --triplet --total 133 ; python test.py  --name  warm5_s1_b24_lr6.2_p0.5_contrast_triplet_133` |
| CE + Contrast + Circle (Long) | 92.19% | 82.07% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 24 --lr 0.08 --name warm5_s1_b24_lr8_p0.5_contrast_circle133 --contrast --circle --total 133 ; python test.py  --name  warm5_s1_b24_lr8_p0.5_contrast_circle133` |
| CE + Contrast + Sphere (Long) | 92.84% | 82.37% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 24 --lr 0.06 --name warm5_s1_b24_lr6_p0.5_contrast_sphere133 --contrast --sphere --total 133 ; python test.py  --name  warm5_s1_b24_lr6_p0.5_contrast_sphere133` |


### Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Prerequisites

- Python 3.6+
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- timm `pip install timm` for Swin-Transformer with Pytorch >1.7.0
- pretrainedmodels via `pip install pretrainedmodels`
- [Optional] apex (for float16) 
- [Optional] [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)

**(Some reports found that updating numpy can arrive the right accuracy. If you only get 50~80 Top1 Accuracy, just try it.)**
We have successfully run the code based on numpy 1.12.1 and 1.13.1 .

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install required packages
```bash
pip install -r requirements.txt
```
- [Optional] You may skip it. Usually it comes with pytorch. Install Torchvision from the source
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optional] You may skip it. Install apex from the source
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .

### Dataset & Preparation

Download [Market1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op) Or use command line:
```bash
pip install gdown 
pip install --upgrade gdown #!!important!!
gdown 0B8-rUzbwVRk0c054eEozWG9COHM
```

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset]( [GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) or ([BaiduYun](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) password: bhbh)) Or use command line:
```bash 
gdown 1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O
```
Our baseline code is not such high on DukeMTMC-reID **Rank@1=64.23%, mAP=43.92%**. Hyperparameters are need to be tuned.

- [Optional] [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) is a generated pedestrian dataset of 128,307 images for training a robust model.

### Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data, e.g., `/home/yourname/Market/pytorch`

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

If you want to use **multiple GPUs**, you are suggested to use DDP (`train_DDP.py`) instead of DP (`train.py`). It is because DP lacks the torch supports and may face some [NaN](https://discuss.pytorch.org/t/nan-loss-with-dataparallel/26501).
You could call `train_DDP.py` by running `DDP.sh`. 
```bash
bash DDP.sh 
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

For some dataset, e.g., MSMT17, there are more than 10 cameras. You need to modify the `prepare.py` and `test.py` to read the double-digit camera ID. Please refer to https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/test_MSMT.py#L234

For some vehicle re-ID datasets. e.g. VeRi, you also need to modify the `prepare.py` and `test.py`.  It has different naming rules.
https://github.com/layumi/Person_reID_baseline_pytorch/issues/107 (Sorry. It is in Chinese)


## Citation
The following paper uses and reports the result of the baseline model. You may cite it in your paper.
```bib
@article{zheng2019joint,
  title={Joint discriminative and generative learning for person re-identification},
  author={Zheng, Zhedong and Yang, Xiaodong and Yu, Zhiding and Zheng, Liang and Yang, Yi and Kautz, Jan},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

The following papers may be the first two to use the bottleneck baseline. You may cite them in your paper.
```bib
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
```bib
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

@article{zheng2020vehiclenet,
  title={VehicleNet: Learning Robust Visual Representation for Vehicle Re-identification},
  author={Zheng, Zhedong and Ruan, Tao and Wei, Yunchao and Yang, Yi and Mei, Tao},
  journal={IEEE Transaction on Multimedia (TMM)},
  year={2020}
}
```

## Related Repos
1. [Pedestrian Alignment Network](https://github.com/layumi/Pedestrian_Alignment) ![GitHub stars](https://img.shields.io/github/stars/layumi/Pedestrian_Alignment.svg?style=flat&label=Star)
2. [2stream Person re-ID](https://github.com/layumi/2016_person_re-ID) ![GitHub stars](https://img.shields.io/github/stars/layumi/2016_person_re-ID.svg?style=flat&label=Star)
3. [Pedestrian GAN](https://github.com/layumi/Person-reID_GAN) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person-reID_GAN.svg?style=flat&label=Star)
4. [Language Person Search](https://github.com/layumi/Image-Text-Embedding) ![GitHub stars](https://img.shields.io/github/stars/layumi/Image-Text-Embedding.svg?style=flat&label=Star)
5. [DG-Net](https://github.com/NVlabs/DG-Net) ![GitHub stars](https://img.shields.io/github/stars/NVlabs/DG-Net.svg?style=flat&label=Star)
6. [3D Person re-ID](https://github.com/layumi/person-reid-3d) ![GitHub stars](https://img.shields.io/github/stars/layumi/person-reid-3d.svg?style=flat&label=Star)
