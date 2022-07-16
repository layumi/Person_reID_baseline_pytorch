import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = '/home/zzd/MSMT17_V1/'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#query
query_path = download_path + 'test/'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for name in open(download_path+'list_query.txt'):
    name = name.split(' ')[0]
    ID = name.split('/')
    src_path = query_path  + name
    dst_path = query_save_path + '/' + ID[0] 
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, dst_path + '/' + os.path.basename(name))

#-----------------------------------------
#gallery
gallery_path = download_path + 'test/'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for name in open(download_path+'list_gallery.txt'):
    name = name.split(' ')[0]
    ID = name.split('/')
    src_path = gallery_path  + name
    dst_path = gallery_save_path + '/' + ID[0]
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, dst_path + '/' + os.path.basename(name))

#---------------------------------------
#train_val
train_path = download_path + 'train/'
val_path = download_path + 'train/'
train_save_path = download_path + '/pytorch/train'
train_all_save_path = download_path + '/pytorch/train_all'
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(train_all_save_path)
    os.mkdir(val_save_path)

for name in open(download_path+'list_train.txt'):
    name = name.split(' ')[0]
    ID = name.split('/')
    src_path = train_path  + name
    dst_path = train_save_path + '/' + ID[0]
    dst_all_path = train_all_save_path + '/' + ID[0]
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    if not os.path.isdir(dst_all_path):
        os.mkdir(dst_all_path)
    copyfile(src_path, dst_path + '/' + os.path.basename(name))
    copyfile(src_path, dst_all_path + '/' + os.path.basename(name))

for name in open(download_path+'list_val.txt'):
    name = name.split(' ')[0]
    ID = name.split('/')
    src_path = val_path  + name
    dst_path = val_save_path + '/' + ID[0]
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    if not os.path.isdir(dst_all_path):
        os.mkdir(dst_all_path)
    copyfile(src_path, dst_path + '/' + os.path.basename(name))
    copyfile(src_path, dst_all_path + '/' + os.path.basename(name))
