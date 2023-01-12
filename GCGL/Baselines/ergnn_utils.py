import random
import torch
import torch.nn as nn

class MF_sampler(nn.Module):
    # sampler for ERGNN MF and MF*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps)

        return self.sampling(ids_per_cls_train, budget, feats)

    def sampling(self,ids_per_cls_train, budget, vecs):
        centers = {i: vecs[ids].mean(0) for i, ids in ids_per_cls_train.items()}
        sim = {i: centers[i].view(1,-1).mm(vecs[ids_per_cls_train[i]].permute(1,0)).squeeze() for i in centers}
        rank = {i: sim[i].sort()[1].tolist() for i in sim}
        ids_selected = {}
        for i,ids in ids_per_cls_train.items():
            nearest = rank[i][0: min(budget, len(ids_per_cls_train[i]))]
            ids_selected[i] = [ids[i] for i in nearest]

        return ids_selected

class CM_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)

        return self.sampling(ids_per_cls_train, budget, feats, d)

    def sampling(self, ids_per_cls_train, budget, vecs, d):
        budget_dist_compute = 1000
        vecs = {cls: vecs[cls].half() for cls in vecs}
        ids_selected = {}
        for i in ids_per_cls_train:
            other_cls_ids = [j for j in ids_per_cls_train if i != j]
            ids_selected0 = random.choices(ids_per_cls_train[i], k=min(budget_dist_compute, len(ids_per_cls_train[i])))
            vecs_0 = vecs[i][ids_selected0]

            dist = []
            for j in other_cls_ids:
                ids_selected1 = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))
                vecs_1 = vecs[j][ids_selected1]
                if len(ids_selected0) < 26 or len(ids_selected1) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0,vecs_1))

            dist_ = torch.cat(dist, dim=-1) # include distance to all the other classes
            n_selected = (dist_ < d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist() # ids after sort
            current_ids_selected = rank[:budget]
            ids_selected[i] = [ids_per_cls_train[i][j] for j in current_ids_selected]

        return ids_selected

class random_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)
        
        return self.sampling(ids_per_cls_train, budget, feats, d)

    def sampling(self,ids_per_cls_train, budget, vecs, d):
        ids_selected = {}
        for i, ids in ids_per_cls_train.items():
            ids_selected[i] = random.sample(ids, min(budget, len(ids)))

        return ids_selected