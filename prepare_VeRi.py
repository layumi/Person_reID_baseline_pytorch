import os
from shutil import copyfile

def copy_file(s, t):
    for root, dirs, files in os.walk(s):
        for name in files:
            copyfile(root+'/'+name,t+'/'+name)

# You only need to change this line to your dataset download path
download_path = './data/VeRi'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/image_all'
train_path = download_path + '/image_train'
gallery_path = download_path + '/image_test'
query_path  = download_path + '/image_query'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

    copy_file(train_path, save_path)
    copy_file(query_path, save_path)
    copy_file(gallery_path,  save_path)

#---------------------------------------
#train
os.mkdir(download_path + '/pytorch')
train_save_path = download_path + '/pytorch/train'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/v' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#query
train_path = query_path
train_save_path = download_path + '/pytorch/query'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/v' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#gallery
train_path = gallery_path
train_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/v' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#train_all
train_path = save_path
train_save_path = download_path + '/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/v' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#train_veri_path =  './data/pytorch/train+veri'
#original_train_save_path = './data/pytorch/train'
#if not os.path.isdir(train_veri_path):
#    os.system('rsync -r %s/ %s/'%(os.path.abspath(train_save_path) , os.path.abspath(train_veri_path) ) )
#    os.system('rsync -r %s/ %s/'%(os.path.abspath(original_train_save_path) , os.path.abspath(train_veri_path) ) )
