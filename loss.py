import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, k=20, mu=0.5):
        super(Loss, self).__init__()
        self.k = k
        self.mu = mu
        
    def forward(self, x1, x2, pair=None):
        if pair is None:
            S = torch.mm(x1.detach(), x2.t().detach())
            S = torch.exp(S*50)
            for i in range(10):
                S /= torch.sum(S, dim=0, keepdims=True)
                S /= torch.sum(S, dim=1, keepdims=True)
            S_max_value, S_max_index = S.max(dim=1)
            index = S_max_value > self.mu
            pair = torch.stack([torch.nonzero(index)[:, 0], S_max_index[index]], dim=1)
        
        S = torch.mm(x1[pair[:, 0]], x2.t())
        index = torch.cat([pair[:, 1].view(-1, 1), S.topk(k=self.k+1).indices[:, 1:]], dim=1)
        S_val = torch.gather(S, 1, index)
        S_val = F.log_softmax(S_val, dim=1)
        loss = -S_val[:, 0].mean()
        return loss
