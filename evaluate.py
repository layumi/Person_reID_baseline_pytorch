import scipy.io
import torch
import numpy as np

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = gf*query
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    index = index[0:2000]
    #print(index[0:5])
    # good index
    good_index = [i for i,x in enumerate(gl) if (x==ql and gc[i]!=qc )]
    junk_index1 = [i for i,x in enumerate(gl) if (x==-2) ]
    junk_index2 = [i for i,x in enumerate(gl) if (x==ql and gc[i]==qc )]
    junk_index = junk_index2 + junk_index1
    #print(len(junk_index))
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    cmc = torch.IntTensor(len(index)).zero_()
    if not  good_index:   # if empty
        cmc[0] = -1
        return cmc
    junk_count = 0
    for i in range(len(index)):
        right = index[i] in good_index
        if right:   # if not empty
            cmc[(i-junk_count):]=1
            break
        if junk_index: #not empty
            junk = index[i] in junk_index
            if junk:
                 junk_count = junk_count+1
    return cmc


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = torch.LongTensor(result['query_cam'])[0]
query_label = torch.LongTensor(result['query_label'])[0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = torch.LongTensor(result['gallery_cam'])[0]
gallery_label = torch.LongTensor(result['gallery_label'])[0]

CMC = torch.IntTensor()
#print(query_label)
for i in range(len(query_label)):
    CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = torch.cat((CMC,CMC_tmp.view(1,len(CMC_tmp))), 0)
    print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC.mean(dim=0) #average CMC
print('top1:%f top5:%f top10:%f'%(CMC[0],CMC[4],CMC[9]))
