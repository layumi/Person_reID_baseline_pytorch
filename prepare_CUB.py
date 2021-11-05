import os
from shutil import copyfile

# 1-100 train
# 101-200 test
os.system('wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz')
os.system('tar -zxvf CUB_200_2011.tgz')
os.system('cd ./CUB_200_2011/images')
os.system('mkdir train_all')
os.system('mkdir test')
os.system('mv 0* train_all')
os.system('mv ./100.Brown_Pelican train_all')
os.system('mv 1* test')
os.system('mv ./200.Common_Yellowthroat/ test')

#---------------------------------------
#train_val
download_path = './CUB_200_2011'
train_path = download_path + '/images/train_all'
train_save_path = download_path + '/images/train'
val_save_path = download_path + '/images/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for r, subdirs,f in os.walk(train_path, topdown=True):
    for sub in subdirs:
        for root, dirs, files in os.walk(train_path+'/'+sub, topdown=True):
            for name in files:
                if not name[-3:]=='jpg':
                    continue
                src_path = train_path + '/' + sub  + '/' + name
                dst_path = train_save_path + '/' + sub
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    dst_path = val_save_path + '/'+ sub   #first image is used as val image
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)
