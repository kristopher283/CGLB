import random

import torch.nn as nn
import numpy as np
import torch
from torch.optim import Adam
import dgl
from dgllife.utils import Meter
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

from .ergnn_utils import CM_sampler, MF_sampler, random_sampler
from GCGL.utils import collate_molgraphs

samplers = {'CM': CM_sampler(plus=False), 'CM_plus': CM_sampler(plus=True), 'MF': MF_sampler(plus=False), 'MF_plus': MF_sampler(plus=True), 'random': random_sampler(plus=False)}

def predict(args, model, bg):
    node_feats = bg.ndata[args['node_data_field']].cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata[args['edge_data_field']].cuda()
        return model(bg, node_feats, edge_feats)

    return model(bg, node_feats)

def predict_feats(args, model, bg):
    node_feats = bg.ndata[args['node_data_field']].cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata[args['edge_data_field']].cuda()
        _, _, raw_feats, hidden_feats = model(bg, node_feats, edge_feats, return_feats=True)

        return raw_feats.detach(), hidden_feats.detach()

    _, _, raw_feats, hidden_feats = model(bg, node_feats, return_feats=True)

    return raw_feats.detach(), hidden_feats.detach()

class NET(torch.nn.Module):
    """
    Bare model baseline for GCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """

    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.optimizer = Adam(model.parameters(), lr=args['lr'])
        self.sampler = samplers[args['dce_args']['sampler']]
        # setup memories
        self.current_task = -1
        self.buffer_graphs = []
        self.budget = int(args['dce_args']['budget'])
        self.d_CM = args['dce_args']['d'] # d for CM sampler of ERGNN
        self.aux_g = None

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data_loader, loss_criterion, task_i, args):
        """
        The method for learning the given tasks under the task-IL setting with multi-label classification datasets.

        :param data_loader: The data loader for mini-batch training.
        :param loss_criterion: The loss function.
        :param task_i: Index of the current task.
        :param args: Same as the args in __init__().

        """

        self.net.train()
        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # Mask non-existing labels
            loss = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss[:,task_i].mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)

        train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_tskIL_multicls(self, data_loader, loss_criterion, task_i, args):
        """
        The method for learning the given tasks under the task-IL setting with multi-class classification datasets.

        :param data_loader: The data loader for mini-batch training.
        :param loss_criterion: The loss function.
        :param task_i: Index of the current task.
        :param args: Same as the args in __init__().

        """
        # task Il under multi-class setting
        self.net.train()
        clss = args['tasks'][task_i]
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            for i, c in enumerate(clss):
                labels[labels == c] = i
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def observe_clsIL(self, data_loader, loss_criterion, task_i, args, prev_model, last_epoch=False):
        """
        The method for learning the given tasks under the class-IL setting with multi-class classification datasets.

        :param data_loader: The data loader for mini-batch training.
        :param loss_criterion: The loss function.
        :param task_i: Index of the current task.
        :param args: Same as the args in __init__().
        :param prev_model: The model obtained after learning the previous task.

        """

        self.net.train()
        clss = []
        for tid in range(task_i + 1):
            clss.extend(args['tasks'][tid])

        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits, _ = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            for i, c in enumerate(clss):
                labels[labels == c] = i
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()

            # sample from the buffer
            if task_i > 0:
                n = logits.shape[0]
                # sample the same number of graphs as the original loss
                batch_data = random.choices(self.buffer_graphs, k=min(n, len(self.buffer_graphs)))
                n_buffer = len(batch_data)
                beta = n_buffer / (n_buffer + n)

                smiles, bg, labels, masks = collate_molgraphs(batch_data)

                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                logits, _ = predict(args, self.net, bg)

                # class balance
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                for i, c in enumerate(clss):
                    labels[labels == c] = i

                dce_loss = 0
                if prev_model is not None:
                    # If there is a previous model, then we get the previous model's logits to calculate the distillation loss.
                    prev_logits, _ = predict(args, prev_model, bg)
                    for oldt in range(task_i):
                        dist_logits = logits[:, oldt]
                        dist_target = prev_logits[:, oldt]
                        step_dce_loss = nn.CosineEmbeddingLoss()(dist_logits.reshape(1, -1), dist_target.reshape(1, -1), torch.ones(1).to(f"cuda:{args['gpu']}"))
                        dce_loss += step_dce_loss

                loss_aux = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
                loss = beta * loss + (1 - beta) * (loss_aux + dce_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if last_epoch:
            # prepare the graph features for the replay
            graphs_per_cls = {}
            raw_feats_per_cls = {}
            hidden_feats_per_cls = {}
            with torch.no_grad():
                for batch_id, batch_data in enumerate(data_loader[task_i]):
                    smiles, bg, labels, masks = batch_data
                    bg = bg.to(f"cuda:{args['gpu']}")
                    labels, masks = labels.cuda(), masks.cuda()
                    # TODO: verify if we need to pool it to get the graph feature
                    raw_feats, hidden_feats = predict_feats(args, self.net, bg)
                    bg = dgl.unbatch(bg)

                    for cls in args['tasks'][task_i]:
                        ids = torch.nonzero(labels == cls).squeeze().tolist()
                        for idx in ids:
                            smile = smiles[idx]
                            g = bg[idx]
                            label = labels[idx]
                            mask = masks[idx]

                            graphs_per_cls[cls] = graphs_per_cls.get(cls, [])
                            graphs_per_cls[cls].append([smile, g, label, mask])

                            raw_feats_per_cls[cls] = raw_feats_per_cls.get(cls , [])
                            raw_feats_per_cls[cls].append(raw_feats[idx])

                            hidden_feats_per_cls[cls] = hidden_feats_per_cls.get(cls , [])
                            hidden_feats_per_cls[cls].append(hidden_feats[idx])
            
            for cls in args['tasks'][task_i]:
                raw_feats_per_cls[cls] = torch.stack(raw_feats_per_cls[cls], dim=0) # shape N x F
                hidden_feats_per_cls[cls] = torch.stack(hidden_feats_per_cls[cls], dim=0) # shape N x F
            
            ids_per_cls_train = {cls: list(range(len(graphs))) for cls, graphs in graphs_per_cls.items()}

            # sample and store ids from current task
            # store only once for each task
            sampled_ids_per_cls = self.sampler(ids_per_cls_train, self.budget, raw_feats_per_cls, hidden_feats_per_cls, self.d_CM) 
            for cls, sampled_ids in sampled_ids_per_cls.items():
                for idx in sampled_ids:
                    self.buffer_graphs.append(graphs_per_cls[cls][idx])