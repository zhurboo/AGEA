import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN_layer(nn.Module):
    def __init__(self, size_v):
        super(GCN_layer, self).__init__()
        self.size_v = size_v

    def forward(self, x, edge, edge_weight):
        self.size_v = x.size(0)
        edge_j, _, edge_i = edge
        if edge_weight is None:
            edge_weight = torch.ones(edge_i.size(0), device=x.device)
        deg = torch.zeros(self.size_v, device=x.device)
        deg.scatter_add_(0, edge_i, edge_weight)
        w = edge_weight/deg[edge_i]
        m = torch.sparse.FloatTensor(edge[[2, 0]], w, (self.size_v, self.size_v))
        x = torch.sparse.mm(m, x)
        x = F.normalize(F.relu(x), dim=1, p=2)
        return x

class GCN_V2E2V_layer(nn.Module):
    def __init__(self, size_v, size_e):
        super(GCN_V2E2V_layer, self).__init__()
        self.size_v = size_v
        self.size_e = size_e
    
    def v2e(self, x_v, edge):
        edge_j, edge_i, _ = edge
        deg = torch.zeros(self.size_e, device=x_v.device)
        deg.scatter_add_(0, edge_i, torch.ones(edge_i.size(0), device=x_v.device))
        w = deg.pow(-1)[edge_i]
        m = torch.sparse.FloatTensor(edge[[1, 0]], w, (self.size_e, self.size_v))
        x_e = torch.sparse.mm(m, x_v)
        x_e = F.relu(x_e)
        return x_e
    
    def e2v(self, x_e, edge):
        edge_i, edge_j, _ = edge
        deg = torch.zeros(self.size_v, device=x_e.device)
        deg.scatter_add_(0, edge_i, torch.ones(edge_i.size(0), device=x_e.device))
        w = deg.pow(-1)[edge_i]
        m = torch.sparse.FloatTensor(edge[[0, 1]], w, (self.size_v, self.size_e))
        x_v = torch.sparse.mm(m, x_e)
        x_v = F.relu(x_v)
        return x_v
    
    def forward(self, x, edge):
        x_e = self.v2e(x, edge)
        x_v = F.normalize(self.e2v(x_e, edge), dim=1, p=2)
        return x_v

    
class GCN(nn.Module):
    def __init__(self, size_v, size_e):
        super(GCN, self).__init__()
        self.gcn_layer = GCN_layer(size_v)
        self.gcn_v2e2v_layer = GCN_V2E2V_layer(size_v, size_e)

    def forward(self, x, edge, edge_weight):
        output = [x]
        output.append(self.gcn_v2e2v_layer(x, edge))
        for i in range(2):
            x = self.gcn_layer(x, edge, edge_weight)
            output.append(x)
        output = torch.cat(output, dim=1)
        output = F.normalize(output, dim=1, p=2)
        return output
    
    
class AGEA(nn.Module):
    def __init__(self, size_v1, size_e1, size_v2, size_e2):
        super(AGEA, self).__init__()
        self.gcn1 = GCN(size_v1, size_e1)
        self.gcn2 = GCN(size_v2, size_e2)
        

    def forward(self, x1, x2, edge1, edge2, edge_weight1=None, edge_weight2=None):
        x1 = self.gcn1(x1, edge1, edge_weight1)
        x2 = self.gcn2(x2, edge2, edge_weight2)
        return x1, x2
