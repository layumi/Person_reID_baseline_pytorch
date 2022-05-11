import os
from shutil import copyfile

def copy_file(s, t):
    for root, dirs, files in os.walk(s):
        for name in files:
            copyfile(root+'/'+name,t+'/'+name)

# You only need to change this line to your dataset download path
download_path = './data/VehicleID_V1.0/'

if not os.path.isdir(download_path):
    print('please change the download_path')

#---------------------------------------
#train_all
train_path = download_path + '/image'
train_save_path = download_path + '/pytorch/train_test'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

    fname = './data/VehicleID_V1.0/attribute/img2vid.txt'
    with open(fname) as fp:
        for i, line in enumerate(fp):
            name, label = line.split(' ')
            name = name + '.jpg'
            ID  = int(label)
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/p%d'%ID
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            print(src_path, dst_path)
            copyfile( src_path, dst_path+'/'+name)

#---------------------------------------
#train
train_list = []
train_only_save_path = download_path + '/pytorch/train'
if not os.path.isdir(train_only_save_path):
    os.mkdir(train_only_save_path)
    with open(download_path+'train_test_split/train_list.txt', 'r') as f:
        for name in f:
            name = name.replace('\n','')
            train_ID = name.split(' ')
            train_ID = int(train_ID[1])
            if not train_ID in train_list:
                train_list.append(train_ID)

        print(len(train_list))
        for ID in train_list:
            os.system('rsync -r %s/p%d %s'%( train_save_path, ID, train_only_save_path))

#---------------------------------------
#val800
for num in [800,1600,2400]:
    val_list = []
    query_save_path = download_path + '/pytorch/query%d'%num
    gallery_save_path = download_path + '/pytorch/gallery%d'%num
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
        os.mkdir(gallery_save_path)
    with open(download_path+'train_test_split/test_list_%d.txt'%num, 'r') as f:
            for name in f:
                name = name.replace('\n','')
                val_ID = name.split(' ')
                val_name = val_ID[0] + '.jpg'
                val_ID = int(val_ID[1])
                src_path = train_path + '/' + val_name
                if val_ID not in val_list:
                    val_list.append(val_ID)
                    dst_path = gallery_save_path + '/p%d'%val_ID #For VehicleID QueryNumber > Gallery
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile( src_path, dst_path+'/'+val_name)
                else:
                    dst_path = query_save_path + '/p%d'%val_ID
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile( src_path, dst_path+'/'+val_name)

