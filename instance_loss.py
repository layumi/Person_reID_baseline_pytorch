import torch
from torch import nn, Tensor
import torch.nn.functional as F

def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-6
    v = v.div(fnorm.expand_as(v))
    return v

class InstanceLoss(nn.Module):
    def __init__(self, gamma = 1) -> None:
        super(InstanceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, feature, label = None) -> Tensor:
        # Dual-Path Convolutional Image-Text Embeddings with Instance Loss, ACM TOMM 2020 
        # https://zdzheng.xyz/files/TOMM20.pdf 
        # using cross-entropy loss for every sample if label is not available. else use given label.
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature*self.gamma, torch.t(normed_feature)) 
        #sim2 = sim1.t()
        if label is None:
            sim_label = torch.arange(sim1.size(0)).cuda().detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label) #+ F.cross_entropy(sim2, sim_label)
        return loss


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(256,))

    criterion = InstanceLoss()
    instance_loss = criterion(feat, lbl)

    print(instance_loss)
