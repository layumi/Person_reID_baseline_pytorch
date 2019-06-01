import os
from shutil import copyfile
import numpy as np


#http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip
download_path = '../VIPeR'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)


#train_all
ID_list = []
train_path = download_path + '/cam_a'
train_save_path = download_path + '/pytorch/all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='bmp':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            ID_list.append(ID[0])
        copyfile(src_path, dst_path + '/' + 'ca_'+name)


train_path = download_path + '/cam_b'
train_save_path = download_path + '/pytorch/all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='bmp':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + 'cb_'+name)

np.random.seed(0)

for ii in range(5):
    index = np.random.permutation(632)
    test_save_path = download_path + '/test%d'%ii
    if not os.path.isdir(test_save_path):
        os.mkdir(test_save_path)
        os.mkdir(test_save_path + '/query')
        os.mkdir(test_save_path + '/gallery')

    for i in range(316):
        dir_name = ID_list[i]
        src_path_1 = train_save_path + '/' + dir_name + '/ca_*.bmp'
        src_path_2 = train_save_path + '/' + dir_name + '/cb_*.bmp'
        dst_path_1 = test_save_path+'/query/'
        dst_path_2 = test_save_path+'/gallery/'
        os.system('cp %s %s'%(src_path_1, dst_path_1))
        os.system('cp %s %s'%(src_path_2, dst_path_2))

