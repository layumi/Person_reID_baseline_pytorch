# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import scipy.io

plt.ion()   # interactive mode
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='2', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
#parser.add_argument('--which_epoch',default=20, type=int, help='0,1,2,3...')
parser.add_argument('--test_dir',default='/home/zzheng/Downloads/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=False, num_workers=4) for x in ['gallery','query']}

#dataset_sizes = len(image_datasets)
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_best.pth')
    network.load_state_dict(torch.load(save_path))
    return network

#####################################################################
#Show result
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
                                                                                            
#imshow(out, 'result')

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    labels = torch.LongTensor() 
    count = 0
    for data in dataloaders:
        count +=1
        print(count)
        img, label = data
        labels = torch.cat((labels,label),0)
        #bs, ncrops, c, h, w = inputs.size()
        #inputs = inputs.view(-1, c, h, w)
        ff = torch.FloatTensor(1,2048).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            f = outputs.data.cpu()
            length = f.size()
            f = f.view(2048,int(length[1]/2048))
            f = f.sum(dim=1)
            f = f.view(1, 2048)
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features,labels

def get_camera(img_path):
    camera_id = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        camera = filename.split('c')[1]
        camera_id.append(int(camera[0]))
    return camera_id

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam = get_camera(gallery_path)
query_cam = get_camera(query_path)

######################################################################
# Load Collected data Trained model
model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
add_block = []
add_block += [nn.Dropout()]
add_block += [nn.Linear(num_ftrs,256)]
add_block += [nn.BatchNorm1d(256)]
add_block += [nn.LeakyReLU(0.1)]
add_block += [nn.Linear(256, 751)]
model_ft.fc = nn.Sequential(*add_block)
model = load_network(model_ft)

# remove the final fc layer
model.fc = nn.Sequential()
# change to test modal
model = model.eval()
if use_gpu:
    model = model.cuda()
#Extract feature
gallery_feature,gallery_label = extract_feature(model,dataloaders['gallery'])
query_feature,query_label = extract_feature(model,dataloaders['query'])
gallery_label = gallery_label - 2

#Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label.numpy(),'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label.numpy(),'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = gf*query
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    # good index
    good_index = [i for i,x in enumerate(gl) if (x==ql and gc[i]!= qc )]
    junk_index1 = [i for i,x in enumerate(gl) if x==-2 ]
    junk_index2 = [i for i,x in enumerate(gl) if (x==ql and gc[i]==qc )]
    junk_index = junk_index1.append(junk_index2)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    cmc = torch.IntTensor(len(index)).zero_()
    if not  good_index:   # if empty
        cmc[0] = -1
        return cmc
    junk_count = 0
    for i in range(len(index)):
        right = [x for x in good_index if x==index[i] ]
        if right:   # if not empty
            cmc[(i-junk_count):]=1
            #print(i)
            break
        if junk_index: #not empty
            junk = [x for x in junk_index if x==index[i] ]
            if junk:
                 junk_count = junk_count+1
    return cmc


######################################################################
CMC = torch.IntTensor()
#print(query_label)
for i in range(len(query_label)):
    CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam,gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = torch.cat((CMC,CMC_tmp.view(1,len(CMC_tmp))), 0)
    
CMC = CMC.float()
CMC = CMC.mean(dim=0) #average CMC
print('top1:%f top5:%f top10:%f'%(CMC[0],CMC[4],CMC[9]))

