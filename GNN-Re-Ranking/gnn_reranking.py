import torch
import numpy as np

import build_adjacency_matrix
import gnn_propagate

from utils import *



def gnn_reranking(X_q, X_g, k1, k2):
    query_num, gallery_num = X_q.shape[0], X_g.shape[0]

    X_u = torch.cat((X_q, X_g), axis = 0)
    original_score = torch.mm(X_u, X_u.t())
    del X_u, X_q, X_g

    # initial ranking list
    S, initial_rank = original_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    
    # stage 1
    A = build_adjacency_matrix.forward(initial_rank.float())   
    S = S * S

    # stage 2
    if k2 != 1:      
        for i in range(2):
            A = A + A.T
            A = gnn_propagate.forward(A, initial_rank[:, :k2].contiguous().float(), S[:, :k2].contiguous().float())
            A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
            A = A.div(A_norm.expand_as(A))                     
    
      
    cosine_similarity = torch.mm(A[:query_num,], A[query_num:, ].t())
    del A, S
    
    L = torch.sort(-cosine_similarity, dim = 1)[1]
    L = L.data.cpu().numpy()
    return L
