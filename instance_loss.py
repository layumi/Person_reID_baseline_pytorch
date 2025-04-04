import torch
from torch import nn, Tensor
import torch.nn.functional as F

def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8
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
        # We assuming that No same ID in the batch. Every one is an unique one. 
            
        if label is not None:# If we have ID, we could merge features of the same class. Then we still treat them as unique ones.  
            unique_ids, inverse_indices = torch.unique(label, return_inverse=True)
            one_hot = torch.nn.functional.one_hot(inverse_indices, num_classes=len(unique_ids)).float()
            sum_features = torch.matmul(one_hot.T, feature)  # (N, feature_dim)
            counts = torch.bincount(inverse_indices)  # (N,)
            feature = sum_features / counts.view(-1, 1)  # (N, feature_dim)
            if len(unique_ids)==1: #handle extreme case.
                return 0

        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature*self.gamma, torch.t(normed_feature))
        sim_label = torch.arange(sim1.size(0))
        loss = F.cross_entropy(sim1, sim_label) 
        return loss


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(32, 512, requires_grad=True))
    lbl = torch.randint(high=750, size=(32,))

    criterion = InstanceLoss(gamma=32)
    instance_loss = criterion(feat, lbl) # assuming some instances are of the same class.
    print(instance_loss)
    instance_loss = criterion(feat) # assuming instance
    print(instance_loss) 
