# UMacau-Person-reID-Practical
[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial)    [![Readme-CN](https://img.shields.io/badge/README-中文-green.svg)](https://zhuanlan.zhihu.com/p/50387521) 

By [Zhedong Zheng](http://zdzheng.xyz/)

This is a [University of Macau](https://www.cis.um.edu.mo/) computer vision practical, authored by Zhedong Zheng.
The practical explores the basis of learning pedestrian features. In this practical, we will learn to build a simple person re-ID system step by step. (8 min read) :+1: **Any suggestion is welcomed.**

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/show.png)

Person re-ID can be viewed as an image retrieval problem. Given one query image in Camera **A**, we need to find the images of the same person in other Cameras. The key of the person re-ID is to find a discriminative representation of the person. Many recent works apply deeply learned models to extract visual features, and achieve the state-of-the-art performance.

We could use this tech to help people. Check the great video by Nvidia. (https://youtu.be/GiZ7kyrwZGQ?t=60)

## Keywords
Person re-identification, 行人重识别, 人の再識別, 보행자 재 식별, Réidentification des piétons, Ri-identificazione pedonale, Fußgänger-Neuidentifizierung, إعادة تحديد المشاة, Re-identificación de peatones

## Ubuntu Usage
**If the machine has installed cuda toolkit and nvidia driver (like our school desktops), you do not need any root permission, like sudo, during our tutorial. If you are using your own machine, I suggest to see this answer to install cuda first. https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu**

Assuming you have an Ubuntu Desktop System,

Press `Ctrl+Alt+T` to open a new terminal. 

The default path is like `\home\user`.

Then you can type `ls` to list all subfolder. 
```bash
ls
```
It may show something like 
```bash
Desktop Images Musics Downloads ...
```

Then you can type `cd XXX' to move to subfolder like 
```bash
cd Downloads # if you download the dataset in the Downloads.
ls # to show all stuff in the \home\user\Downloads
cd ..   #Back to the upper folder. You are at \home\user again.
```

## Windows Usage (Not Recommended)
We do not suggest using Windows considering lower GPU usage and unexpected errors.
If you still want to use Windows, you should keep two points in mind. 
- Path: Ubuntu path is `\home\zzd\` but Windows path is `D://Downloads/` using `/` instead of `\` 
- Multi-thread: Pytorch does not support multiple thread to read the data. Please set `num_workers=0` during trainning and test.
- No Triton or other error: Please remove the `torch.compile` in [this training line](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/train.py#L512) and [this test line](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/test.py#L167).

Please also refer to https://github.com/layumi/Person_reID_baseline_pytorch/issues/34 

## Colab Usage (Not Recommended)
Please refer to https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/colab 

## Prerequisites
- Download my repo
```bash
git clone https://github.com/layumi/Person_reID_baseline_pytorch.git # Download the whole code.
cd Person_reID_baseline_pytorch
```
- Install Pytorch from http://pytorch.org/
- Install required packages
```bash
pip install -r requirements.txt
```
- [Optional] No pip or python?? You may install it without sudo permission by installing miniconda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
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
In most case, we also suport the latest pytorch. We usually recommend to use the latest pytorch. 


## Getting started
Check the Prerequisites. The download links for this practice are:

- Code: [ReID-Baseline](https://github.com/layumi/Person_reID_baseline_pytorch)
- Data: [Market-1501](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)

A quick command line to download Market-1501 is: 
```
pip install gdown
gdown https://drive.google.com/uc\?id\=0B8-rUzbwVRk0c054eEozWG9COHM
```


## Part 1: Training
### Part 1.1: Prepare Data Folder (`python prepare.py`)
You may notice that the downloaded folder is organized as:
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
```
Open and edit the script `prepare.py` in the editor. Change the fifth line in `prepare.py` to your download path, such as `\home\zzd\Download\Market`. Run this script in the terminal.
```bash
python prepare.py
```
We create a subfolder called `pytorch` under the download folder. 
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
│       ├── multi-query/    
```

In every subdir, such as `pytorch/train/0002`, images with the same ID are arranged in the folder.
Now we have successfully prepared the data for `torchvision` to read the data. 

```diff
+ Quick Question. How to recognize the images of the same ID?
```
For Market-1501, the image name contains the identity label and camera id. Check the naming rule at [here](http://www.liangzheng.org/Project/project_reid.html).

For DukeMTMC, you can use my modified `python prepare_Duke.py`.

### Part 1.2: Build Neural Network (`model.py`)
We can use the pretrained networks, such as `AlexNet`, `VGG16`, `ResNet` and `DenseNet`. Generally, the pretrained networks help to achieve a better performance, since it preserves some good visual patterns from ImageNet [1].

In pytorch, we can easily import them by two lines. For example,
```python
from torchvision import models
model = models.resnet50(pretrained=True)
```
You can simply check the structure of the model by:
```python
print(model)
```

But we need to modify the networks a little bit. There are 751 classes (different people) in Market-1501, which is different with 1,000 classes in ImageNet. So here we have changed the model to use our classifier (I have modified it for you, so you do not need to modify the code. Please just take a look).

```python
import torch
import torch.nn as nn
from torchvision import models

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num = 751):   # Check this line. 
        super(ft_net, self).__init__()
        #load the model
        model_ft = models.resnet50(pretrained=True) 
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num) #define our classifier.

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x
```

```diff
+ Quick Question. Why we use AdaptiveAvgPool2d? What is the difference between the AvgPool2d and AdaptiveAvgPool2d?
+ Quick Question. Does the model have parameters now? How to initialize the parameter in the new layer?
```
More details are in `model.py`. You may check it later after you have gone through this practical.

### Part 1.3: Training (`python train.py`)
OK. Now we have prepared the training data and defined model structure.

We can train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of the model.

`--data_dir` the path of the training data, e.g., `/home/yourname/Market/pytorch`

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

If you meet error `python 3.12+ does not support dynamic.`, we could (1) delete the compile operation, which requires dynamic in both training and test code (https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/train.py#L512 and https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/test.py#L167). 
(2) Or creat a new environment with python 3.9 by 
```
conda create --name python39 python=3.9
conda activate python39
```
and then reinstall all mentioned requirements again. 

Open another terminal to see the GPU usage. 
```bash
nvidia-smi # show dense version
pip install gpustat
gpustat # show brief version
```

Now let's look at what we do in the `train.py`.
The first thing is how to read data and their labels from the prepared folder.
Using `torch.utils.data.DataLoader`, we can obtain two iterators `dataloaders['train']` and `dataloaders['val']` to read data and label.
```python
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
```

Here is the main code to train the model.
Yes. It's only about 20 lines. Make sure you can understand every line of the code.
```python
            # Iterate over data.
            for data in dataloaders[phase]:
                # get a batch of inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable, if gpu is used, we transform the data to cuda.
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                #-------- forward --------
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                #-------- backward + optimize -------- 
                # only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
```
```diff
+ Quick Question. Why we need optimizer.zero_grad()? What happens if we remove it?
+ Quick Question. The dimension of the outputs is batchsize*751. Why?
```
Every 10 training epoch, we save a snapshot and update the loss curve.
```python
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
```

## Part 2: Test
### Part 2.1: Extracting feature (`python test.py`)
In this part, we load the network weight (we just trained) to extract the visual feature of every image.
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 60
```
`--gpu_ids` which gpu to run.

`--name` the dir name of the trained model.


`--batchsize` batch size.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.

Let's look at what we do in the `test.py`.
First, we need to import the model structure and then load the weight to the model.
```python
model_structure = ft_net(751)
model = load_network(model_structure)
```
For every query and gallery image, we extract the feature by simply forward the data.
```python
outputs = model(input_img) 
# ---- L2-norm Feature ------
ff = outputs.data.cpu()
fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
ff = ff.div(fnorm.expand_as(ff))
```
```diff
+ Quick Question. Why we flip the test image horizontally when testing? How to fliplr in pytorch?
+ Quick Question. Why we L2-norm the feature?
```

### Part 2.2: Evaluation
Yes. Now we have the feature of every image. The only thing we need to do is matching the images by the feature.
```bash
python evaluate_gpu.py
```
If you get a result, which is much worse than our baseline like only 10% or 20%, please check your numpy version first. Someone has reported numpy issue.
If you get a result, which is close to 0 or 100 or error, you need to check your path setting. Do you give a right test path during `test.py`? 


Let's look what we do in `evaluate_gpu.py`. We sort the predicted similarity score.
```python
query = qf.view(-1,1)
# print(query.shape)
score = torch.mm(gf,query) # Cosine Distance
score = score.squeeze(1).cpu()
score = score.numpy()
# predict index
index = np.argsort(score)  #from small to large
index = index[::-1]
```

Note that there are two kinds of images we do not consider as right-matching images.
* Junk_index1 is the index of mis-detected images, which contain the body parts.

* Junk_index2 is the index of the images, which are of the same identity in the same cameras.

```python
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    # The images of the same identity in different cameras
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # Only part of body is detected. 
    junk_index1 = np.argwhere(gl==-1)
    # The images of the same identity in same cameras
    junk_index2 = np.intersect1d(query_index, camera_index)
```

We can use the function `compute_mAP` to obtain the final result.
In this function, we will ignore the junk_index.
```python
CMC_tmp = compute_mAP(index, good_index, junk_index)
```

## Part 3: A simple visualization (`python demo.py`)
To visualize the result, 
```
python demo.py --query_index 777
```
`--query_index ` which query you want to test. You may select a number in the range of `0 ~ 3367`.

It is similar to the `evaluate.py`. We add the visualization part.
```python
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(10): #Show top-10 images
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green') # true matching
        else:
            ax.set_title('%d'%(i+1), color='red') # false matching
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
```

## Part 4: Your Turn. 
For the assignment, you are free to select any related topics. Here I just give some basic ideas. You do not need to finish all.

- Try different datasets. Market-1501 is a dataset collected at Tsinghua University in summer.

Let's try another dataset called DukeMTMC-reID, which is collected at Duke University in winter.

You may download the dataset at [GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) or ([BaiduYun](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) password: bhbh) or using the following bash. Try it by yourself.
```bash
gdown 1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O
python prepare_Duke.py # please also modify the path. 
```

The dataset is quite similar to Market-1501. You may also check with the state-of-the-art results at [Here](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/leaderboard/README-Duke.md). 

```diff
+ Quick Question. Could we directly apply the model trained on Market-1501 to DukeMTMC-reID? Why?
```

- Try different backbones. https://github.com/layumi/Person_reID_baseline_pytorch/tree/master?tab=readme-ov-file#trained-model

- Try different loss compositions. https://github.com/layumi/Person_reID_baseline_pytorch/tree/master?tab=readme-ov-file#different-losses

- Try verification + identification loss. You may check the code at [Here](https://github.com/layumi/Person-reID-verification).

- Try Triplet Loss.
Triplet loss is another widely-used objective. You may check the code in https://github.com/layumi/Person-reID-triplet-loss. 
I write the code in a similar manner, so let's find what I changed.

## Part5: Other Related Works
- The pedestrian has some specific attributes, e.g., gender, carrying. They can help the feature learning. We annotate the ID-level attributes for Market-1501 and DukeMTMC-reID. You could check [this paper](https://arxiv.org/abs/1703.07220).
![](https://github.com/vana77/DukeMTMC-attribute/blob/master/sample_image.jpg)

- Could we use natural language as query? Check [this paper](https://arxiv.org/abs/1711.05535).
![](https://github.com/layumi/Image-Text-Embedding/blob/master/CUHK-show.jpg)

- Could we use other losses (i.e. contrastive loss) to further improve the performance? Check [this paper](https://arxiv.org/abs/1611.05666). 
![](https://github.com/layumi/2016_person_re-ID/raw/master/paper.jpg)

- Person-reID dataset is not large enough to train a deep-learned network? You may check [this paper](https://arxiv.org/abs/1701.07717) (use GAN to generate more samples) and try some data augmentation method like [random erasing](https://arxiv.org/abs/1708.04896).

![](https://github.com/layumi/Person-reID_GAN/raw/master/fig0.jpg)

- Pedestrian detection is bad? Try [Open Pose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [Spatial Transformer](https://github.com/layumi/Pedestrian_Alignment) to align the images.

![](https://github.com/layumi/Pedestrian_Alignment/raw/master/fig2.jpg)

- Data Limitation? Generate more! [Code](https://github.com/NVlabs/DG-Net)

![](https://github.com/NVlabs/DG-Net/raw/master/NxN.jpg)

- 3D Person Re-identification [Code](https://github.com/layumi/person-reid-3d)
![](https://github.com/layumi/person-reid-3d/blob/master/imgs/demo-1.jpg)

## Answers to Quick Questions
You may check https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/tutorial/Answers_to_Quick_Questions.md

## Star History

If you like this repo, please star it. Thanks a lot!

<img width="884" alt="image" src="https://github.com/user-attachments/assets/c16e8341-8a78-4e62-addc-9af610690e65" />


[![Star History Chart](https://api.star-history.com/svg?repos=layumi/Person_reID_baseline_pytorch&type=Date)](https://star-history.com/#layumi/Person_reID_baseline_pytorch&Date)

## Reference

[1] Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "Imagenet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pp. 248-255. Ieee, 2009.

[2] Zheng, Zhedong, Liang Zheng, and Yi Yang. "Unlabeled samples generated by gan improve the person re-identification baseline in vitro." In Proceedings of the IEEE International Conference on Computer Vision, pp. 3754-3762. 2017.

[3] Zheng, Zhedong, Xiaodong Yang, Zhiding Yu, Liang Zheng, Yi Yang, and Jan Kautz. "Joint discriminative and generative learning for person re-identification." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2138-2147. 2019.

[4] Zheng, Zhedong, Liang Zheng, and Yi Yang. "A discriminatively learned cnn embedding for person reidentification." ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14, no. 1 (2017): 1-20.

[5] Zheng, Zhedong, Liang Zheng, and Yi Yang. "Pedestrian alignment network for large-scale person re-identification." IEEE Transactions on Circuits and Systems for Video Technology 29, no. 10 (2018): 3037-3045.

[6] Zheng, Zhedong, Liang Zheng, Michael Garrett, Yi Yang, and Yi-Dong Shen. "Dual-path convolutional image-text embedding with instance loss." ACM TOMM 2020.
