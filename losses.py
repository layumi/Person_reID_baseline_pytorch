import os
import torch
import yaml
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
import math

def L2Normalization(ff, dim = 1):
     # ff is B*N
     fnorm = torch.norm(ff, p=2, dim=dim, keepdim=True) + 1e-5
     ff = ff.div(fnorm.expand_as(ff))
     return ff

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

# I largely modified the AngleLinear Loss
class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        init.normal_(self.weight.data, std=0.001)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

#https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py#L80
class ArcLinear(nn.Module):
    def __init__(self, in_features, out_features, s=64.0):
        super(ArcLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        init.normal_(self.weight.data, std=0.001)
        self.loss_s = s

    def forward(self, input):
        embedding = input
        nembedding = L2Normalization(embedding, dim=1)*self.loss_s
        _weight = L2Normalization(self.weight, dim=0)
        fc7 = nembedding.mm(_weight)
        output = (fc7, _weight, nembedding)
        return output

class ArcLoss(nn.Module):
    def __init__(self, m1=1.0, m2=0.5, m3 =0.0, s = 64.0):
        super(ArcLoss, self).__init__()
        self.loss_m1 = m1
        self.loss_m2 = m2
        self.loss_m3 = m3
        self.loss_s = s

    def forward(self, input, target):
        fc7, _weight, nembedding = input

        index = fc7.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        zy = fc7[index]
        cos_t = zy/self.loss_s
        t = torch.acos(cos_t)
        t = t*self.loss_m1 + self.loss_m2
        body = torch.cos(t) - self.loss_m3

        new_zy = body*self.loss_s
        diff = new_zy - zy
        fc7[index] += diff
        loss = F.cross_entropy(fc7, target)
        return loss

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss



