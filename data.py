import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Data():
    def __init__(self, path, embs, rate=0.3, seed=1, device='cuda', rev_edge=True):
        self.path = path
        self.rate = rate
        self.seed = seed
        self.embs = embs
        self.device = device
        self.rev_edge = rev_edge
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        file = os.path.join(self.path, str(rate)+'_'+str(seed)+'_'+str(rev_edge)+'.pt')
        if os.path.exists(file):
            self.load(file, self.device)
        else:
            self.process()
            self.save(file)

    def process(self):
        ids_1 = os.path.join(self.path, 'ent_ids_1')
        ids_2 = os.path.join(self.path, 'ent_ids_2')
        names_1 = os.path.join(self.path, 'ent_names_1')
        names_2 = os.path.join(self.path, 'ent_names_2')
        triples_1 = os.path.join(self.path, 'triples_1')
        triples_2 = os.path.join(self.path, 'triples_2')
        pairs = os.path.join(self.path, 'ref_ent_ids')
        embs = {}
        with open(self.embs) as f:
            for line in tqdm.tqdm(f.readlines()):
                line = line.strip().split()
                embs[line[0]] = torch.tensor([float(x) for x in line[1:]])
        x1, edge1, ent_ref1, g1 = self.process_graph(ids_1, names_1, triples_1, embs)
        x2, edge2, ent_ref2, g2 = self.process_graph(ids_2, names_2, triples_2, embs)
        self.x1, self.edge1, self.g1 = x1.to(self.device), edge1.to(self.device), g1
        self.x2, self.edge2, self.g2 = x2.to(self.device), edge2.to(self.device), g2
        pairs = self.process_pair(pairs, ent_ref1, ent_ref2)
        pairs = pairs[torch.randperm(pairs.size(0))]
        self.train_set = pairs[:int(self.rate*pairs.size(0))].to(self.device)
        self.eval_set = pairs[int(self.rate*pairs.size(0)):int((self.rate+0.05)*pairs.size(0))].to(self.device)
        self.test_set = pairs[int(self.rate*pairs.size(0)):].to(self.device)
    
    def process_g(self, edges):
        g = {}
        for h, r, t in edges.t().numpy():
            if h not in g.keys():
                g[h] = set()
            g[h].add(t)
            if t not in g.keys():
                g[t] = set()
            g[t].add(h)
        for e in g.keys():
            g[e] = sorted(list(g[e]))
        return g
        
    def process_graph(self, ids, names, triples, embs):
        with open(ids, 'r') as f:
            ents = [int(line.strip().split('\t')[0]) for line in f.readlines()]
        ent_ref = {ent:i for i, ent in enumerate(ents)}
        x = [None for i in range(len(ents))]
        with open(names, 'r') as f:
            for line in f.readlines():
                try:
                    ent, name = line.strip().split('\t')
                except:
                    ent = line.strip()
                    name = ''
                ent_x = []
                for word in name.split():
                    word = word.lower()
                    if word in embs.keys():
                        ent_x.append(embs[word])
                if len(ent_x) > 0:
                    x[ent_ref[int(ent)]] = torch.stack(ent_x, dim=0).mean(dim=0)
                else:
                    x[ent_ref[int(ent)]] = torch.rand(300)-0.5
        x = torch.stack(x, dim=0).contiguous()
        x = F.normalize(x, dim=1, p=2)
        with open(triples, 'r') as f:
            edges = [list(map(int, line.strip().split('\t'))) for line in f.readlines()]
        rels = set([r for _, r, _ in edges])
        rel_ref = {rel:i for i, rel in enumerate(rels)}
        edges = torch.LongTensor([[ent_ref[h], rel_ref[r], ent_ref[t]] for h, r, t in edges]).t().contiguous()
        g = self.process_g(edges)
        if self.rev_edge:
            rel_size = edges[1].max()+1
            edges_rev = torch.stack([edges[2], edges[1]+rel_size, edges[0]], dim=0)
            edges = torch.cat([edges, edges_rev], dim=1)
        return x, edges, ent_ref, g

    def process_pair(self, pairs, ent_ref1, ent_ref2):
        with open(pairs, 'r') as f:
            pairs = [list(map(int, line.strip().split('\t'))) for line in f.readlines()]
        pairs = torch.LongTensor([[ent_ref1[e1], ent_ref2[e2]] for e1, e2 in pairs])
        return pairs
    
    def save(self, file):
        data = [self.x1.cpu(), self.x2.cpu(), self.edge1.cpu(), self.g1, self.edge2.cpu(), self.g2,
                self.train_set.cpu(), self.eval_set.cpu(), self.test_set.cpu()]
        torch.save(data, file)
    
    def load(self, file, device):
        x1, x2, edge1, g1, edge2, g2, train_set, eval_set, test_set = torch.load(file)
        self.x1, self.edge1, self.g1 = x1.to(device), edge1.to(device), g1
        self.x2, self.edge2, self.g2 = x2.to(device), edge2.to(device), g2
        self.train_set, self.eval_set, self.test_set = train_set.to(device), eval_set.to(device), test_set.to(device)
        
