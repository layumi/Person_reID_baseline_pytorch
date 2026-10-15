import os
from prepare_helpers import copy_to_identity_folder

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
    copy_to_identity_folder(query_path, query_save_path, name)

#-----------------------------------------
#gallery
gallery_path = download_path + 'test/'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for name in open(download_path+'list_gallery.txt'):
    name = name.split(' ')[0]
    copy_to_identity_folder(gallery_path, gallery_save_path, name)

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
    copy_to_identity_folder(train_path, train_save_path, name)
    copy_to_identity_folder(train_path, train_all_save_path, name)

for name in open(download_path+'list_val.txt'):
    name = name.split(' ')[0]
    copy_to_identity_folder(val_path, val_save_path, name)
    copy_to_identity_folder(val_path, train_all_save_path, name)
