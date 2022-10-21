import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def cal_edge_weight1(g1, g2, edge_weight1_index, edge_weight1_w1, index, S, lam, q):
    edge_weight_map1 = {}
    for i in index:
        j = edge_weight1_index[i]
        w1 = max(edge_weight1_w1[i], 0.2)
        S_nbr = S[g1[i], :][:, g2[j]]
        if S_nbr.shape[1] == 1:
            w2s = S_nbr[:, 0]
        else:
            S_nbr_top2_index = np.partition(S_nbr, -2, axis=1)[:, -2:]
            w2s = S_nbr_top2_index[:, 1]-S_nbr_top2_index[:, 0]
        for _, k in enumerate(g1[i]):
            w2 = max(w2s[_], lam)
            edge_weight_map1[(i, k)] = w1*w2
    q.put(edge_weight_map1)

    
def cal_edge_weight2(g1, g2, edge_weight2_index, edge_weight2_w1, index, S, lam, q):
    edge_weight_map2 = {}
    for i in index:
        j = edge_weight2_index[i]
        w1 = max(edge_weight2_w1[i], lam)
        S_nbr = S[g1[j], :][:, g2[i]]
        if S_nbr.shape[0] == 1:
            w2s = S_nbr[0, :]
        else:
            S_nbr_top2_index = np.partition(S_nbr, -2, axis=0)[-2:, :]
            w2s = S_nbr_top2_index[1, :]-S_nbr_top2_index[0, :]
        for _, k in enumerate(g2[i]):
            w2 = max(w2s[_], lam)
            edge_weight_map2[(i, k)] = w1*w2
    q.put(edge_weight_map2)

    
def cal_edge_weight(x1, x2, edge1, g1, edge2, g2, lam):
    S = torch.mm(x1, x2.t())
    S = torch.exp(S*50)
    for i in range(10):
        S /= torch.sum(S, dim=0, keepdims=True)
        S /= torch.sum(S, dim=1, keepdims=True)
    S_numpy = S.cpu().numpy()
    edge_weight1_w1_top2_value, edge_weight1_w1_top2_index = S.topk(k=2, dim=1)
    edge_weight1_w1 = (edge_weight1_w1_top2_value[:, 0]-edge_weight1_w1_top2_value[:, 1]).cpu().numpy()
    edge_weight1_index = edge_weight1_w1_top2_index[:, 0].cpu().numpy()
    edge_weight2_w1_top2_value, edge_weight2_w1_top2_index = S.topk(k=2, dim=0)
    edge_weight2_w1 = (edge_weight2_w1_top2_value[0, :]-edge_weight2_w1_top2_value[1, :]).cpu().numpy()
    edge_weight2_index = edge_weight2_w1_top2_index[0, :].cpu().numpy()
    
    q1 = multiprocessing.Queue()
    jobs1 = []
    work_num = 8
    for i in range(work_num):
        if i == work_num-1:
            index = range(len(edge_weight1_index)//8*i,len(edge_weight1_index))
        else:
            index = range(len(edge_weight1_index)//8*i,len(edge_weight1_index)//8*(i+1))
        p = multiprocessing.Process(target=cal_edge_weight1, 
                                    args=(g1, g2, edge_weight1_index, edge_weight1_w1, index, S_numpy, lam, q1))
        p.start()
        jobs1.append(p)
    q2 = multiprocessing.Queue()
    jobs2 = []
    work_num = 8
    for i in range(work_num):
        if i == work_num-1:
            index = range(len(edge_weight2_index)//8*i,len(edge_weight2_index))
        else:
            index = range(len(edge_weight2_index)//8*i,len(edge_weight2_index)//8*(i+1))
        p = multiprocessing.Process(target=cal_edge_weight2, 
                                    args=(g1, g2, edge_weight2_index, edge_weight2_w1, index, S_numpy, lam, q2))
        p.start()
        jobs2.append(p)
        
    edge_weight_map1 = {}
    [edge_weight_map1.update(q1.get()) for _ in jobs1]
    edge_weight1 = []
    for h, _, t in edge1.t().cpu().numpy():
        edge_weight1.append(max(edge_weight_map1[(h, t)], edge_weight_map1[(t, h)]))
    edge_weight1 = torch.tensor(edge_weight1, device=x1.device, dtype=torch.float32)
    edge_weight_map2 = {}
    [edge_weight_map2.update(q2.get()) for _ in jobs2]
    edge_weight2 = []
    for h, _, t in edge2.t().cpu().numpy():
        edge_weight2.append(max(edge_weight_map2[(h, t)], edge_weight_map2[(t, h)]))
    edge_weight2 = torch.tensor(edge_weight2, device=x2.device, dtype=torch.float32)
    return edge_weight1, edge_weight2


def get_hits_COS(x1, x2, pair, Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.mm(x1[pair[:, 0]], x2[pair[:, 1]].t())
    Hks = []
    for k in Hn_nums:
        pred_topk= S.topk(k)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        Hks.append(round(Hk*100, 2))
    rank = torch.where(S.sort(descending=True)[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = round((1/(rank+1)).mean().item(), 3)
    return Hks+[MRR]


def get_hits_Sinkhorn(x1, x2, pair, Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.mm(x1[pair[:, 0]], x2[pair[:, 1]].t())
    S = torch.exp(S*50)
    for i in range(10):
        S /= torch.sum(S, dim=0, keepdims=True)
        S /= torch.sum(S, dim=1, keepdims=True)
    Hks = []
    for k in Hn_nums:
        pred_topk= S.topk(k)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        Hks.append(round(Hk*100, 2))
    rank = torch.where(S.sort(descending=True)[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = round((1/(rank+1)).mean().item(), 3)
    return Hks+[MRR]
