import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AGEA
from data import Data
from loss import Loss
from utils import setup_seed, cal_edge_weight, get_hits_COS, get_hits_Sinkhorn


    
def train(model, criterion, optimizer, data, unspervised=False):
    model.train()
    x1, x2 = model(data.x1, data.x2, data.edge1, data.edge2, data.edge_weight1, data.edge_weight2)
    if unspervised:
        loss = criterion(x1, x2)
    else:
        loss = criterion(x1, x2, data.train_set)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    

def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1, x2 = model(data.x1, data.x2, data.edge1, data.edge2, data.edge_weight1, data.edge_weight2)
    return x1, x2


def test(x1, x2, test_set, name):
    with torch.no_grad():
        Cos = get_hits_COS(x1, x2, test_set)
        Sinkhorn = get_hits_Sinkhorn(x1, x2, test_set)
        print(f'{name} Cos: {Cos}, Sinkhorn: {Sinkhorn}')

    
def test_all(x1, x2, data, unspervised=False):
    if not unspervised:
        test(x1, x2, data.train_set, 'Train')
    test(x1, x2, data.eval_set, 'Eval ')
    test(x1, x2, data.test_set, 'Test ')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--data", default="data/dbp15k_zh_en")
    parser.add_argument("--embs", default="data/glove.6B.300d.txt")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--test_epoch", type=int, default=10)
    parser.add_argument("--adaptive", action="store_true", default=False)
    parser.add_argument("--weight_epoch", type=int, default=15)
    parser.add_argument("--unspervised", action="store_true", default=False)
    args = parser.parse_args()

    setup_seed(args.seed)
    data = Data(args.data, args.embs, args.rate, args.seed, args.device)
    if args.unspervised:
        data.test_set = torch.cat([data.train_set, data.test_set])
    data.x1.requires_grad_(), data.x2.requires_grad_()
    data.edge_weight1, data.edge_weight2 = None, None
    test_all(data.x1, data.x2, data, args.unspervised)
    size_e1, size_e2 = data.x1.size(0), data.x2.size(0)
    size_r1, size_r2 = max(data.edge1[1])+1, max(data.edge2[1])+1
    model = AGEA(size_e1, size_r1, size_e2, size_r2).to(args.device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x1, data.x2])))
    criterion = Loss(args.k)
    maxx = 0
    best_h1, best_x1, best_x2 = 0, None, None
    for epoch in range(args.epoch):
        loss = train(model, criterion, optimizer, data, args.unspervised)
        print(f'----------Epoch: {epoch+1}/{args.epoch}, Loss: {loss:.4f}----------\r', end='')
        x1, x2 = get_emb(model, data)
        if args.adaptive and (epoch+1)%args.weight_epoch == 0:
            data.edge_weight1, data.edge_weight2 = cal_edge_weight(x1, x2, data.edge1, data.g1, data.edge2, data.g2, args.lam)
        h1 = get_hits_COS(x1, x2, data.eval_set)[0]
        if h1 > best_h1:
            best_h1 = h1
            best_x1, best_x2 = x1.cpu(), x2.cpu()
        if (epoch+1)%args.test_epoch == 0:
            print()
            test_all(x1, x2, data, args.unspervised)
    print('----------Final Results----------')
    x1, x2 = best_x1.to(args.device), best_x2.to(args.device)
    test_all(x1, x2, data, args.unspervised)
