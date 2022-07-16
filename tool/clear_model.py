import os

root = '../model/'
nn = []
for f in os.listdir(root):
    if f == '.gitkeep':
        continue
    for ff in os.listdir(root+f):
        if ff[0:3] == 'net':
            if ff[5] =='a':
                continue
            else:
                path = root+f+'/'+ff
                print(path)
                os.remove(path)
