import random

import numpy as np
import torch
from torch.optim import Adam
import dgl
from dgllife.utils import Meter
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

from .ergnn_utils import CM_sampler, MF_sampler, random_sampler
from GCGL.utils import collate_molgraphs

samplers = {'CM': CM_sampler(plus=False), 'CM_plus': CM_sampler(plus=True), 'MF': MF_sampler(plus=False), 'MF_plus': MF_sampler(plus=True), 'random': random_sampler(plus=False)}
K_SAMPLES = 10


def predict(args, model, bg, return_node_feats=False):
    node_feats = bg.ndata[args['node_data_field']].cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata[args['edge_data_field']].cuda()
        return model(bg, node_feats, edge_feats)

    return model(bg, node_feats, return_node_feats=return_node_feats)

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
        self.sampler = samplers[args['sl_args']['sampler']]
        # setup memories
        self.current_task = -1
        self.buffer_graphs = []
        self.budget = int(args['sl_args']['budget'])
        self.max_size = int(args['sl_args']['max_size'] * args['n_cls'] * self.budget)
        self.d_CM = args['sl_args']['d'] # d for CM sampler of SL
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
                logits, feats = predict(args, self.net, bg, return_node_feats=True)

                # class balance
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                for i, c in enumerate(clss):
                    labels[labels == c] = i

                loss_aux = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
                # loss = beta * loss + (1 - beta) * loss_aux
                loss_sl = self.get_sl_loss(prev_model, bg, feats, args)
                loss = beta * loss + (1 - beta) * (loss_aux + loss_sl)

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
            
            # when the buffer graphs has passed the max_size
            if len(self.buffer_graphs) > self.max_size:
                print(f"Current size of replay buffer {len(self.buffer_graphs)} > max_size")
                buffer_size = len(self.buffer_graphs)
                ids_per_cls_buffer = {}
                for cls in clss:
                    ids_per_cls_buffer[cls] = [idx for idx, graph in enumerate(self.buffer_graphs) if (graph[2] == cls).sum() > 0]

                removed = []
                while buffer_size > self.max_size:
                    largest_cls = max(ids_per_cls_buffer, key=lambda cls: len(ids_per_cls_buffer[cls]))
                    _removed = random.choice(ids_per_cls_buffer[largest_cls])
                    ids_per_cls_buffer[largest_cls].remove(_removed)
                    removed.append(_removed)
                    buffer_size -= 1
                
                if len(removed) != len(set(removed)):
                    import ipdb; ipdb.set_trace()
                
                # actually remove them from self.buffer_graphs
                removed_graphs = [self.buffer_graphs[idx] for idx in removed]
                for g in removed_graphs:
                    self.buffer_graphs.remove(g)
                

    def get_sl_loss(self, prev_model, aux_g, cur_feats, args):
        structure_loss = 0
        if prev_model is not None:
            # If there is a previous model, then we get the previous model's logits to calculate the distillation loss.
            prev_logits, prev_feats = predict(args, prev_model, aux_g, return_node_feats=True)

            # feat_src, _ = expand_as_pair(aux_g.srcdata['feat'])
            # aux_g.srcdata['h'] = feat_src
            # aux_g.apply_edges(lambda edges: {'se': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            # soft_edges = aux_g.edata.pop('se')

            adj_matrix = aux_g.adj()
            rand_k_node_samples = random.sample(range(0, aux_g.num_nodes()), K_SAMPLES)
            for node_idx in rand_k_node_samples:
                # For the old (previous task) model.
                # Get the different in term of features between the target node and its neighbor nodes. (This aims to extract the
                # structure information between the node and its neighbors).
                ref_neighbor_nodes = prev_feats[adj_matrix[node_idx].to_dense().bool()]
                # ref_neighbor_nodes = soft_edges.unsqueeze(1) * prev_feats
                if ref_neighbor_nodes.numel() > 0:
                    ref_neighbors_feat = ref_neighbor_nodes.sum(dim=0)
                    ref_diff_vector = prev_feats[node_idx] - ref_neighbors_feat
                else:
                    ref_diff_vector = None

                # For the current model.
                # Get the different in term of features between the target node and its neighbor nodes. (This aims to extract the
                # structure information between the node and its neighbors).
                cur_neighbor_nodes = cur_feats[adj_matrix[node_idx].to_dense().bool()]
                # cur_neighbor_nodes = soft_edges.unsqueeze(1) * cur_feats
                if cur_neighbor_nodes.numel() > 0:
                    cur_neighbors_feat = cur_neighbor_nodes.sum(dim=0)
                    cur_diff_vector = cur_feats[node_idx] - cur_neighbors_feat
                else:
                    cur_diff_vector = None

                if ref_diff_vector is not None and cur_diff_vector is not None:
                    if (ref_diff_vector == cur_diff_vector).all():
                        # Skip if two vectors are similar.
                        continue
                    
                    # Calculate the difference (similarity) of the learned structure information between the old model and the
                    # current model.
                    step_structure_loss = torch.nn.CosineEmbeddingLoss()(torch.unsqueeze(ref_diff_vector, dim=0),
                                                                    torch.unsqueeze(cur_diff_vector, dim=0),
                                                                    torch.ones(1).cuda(args['gpu']))
                    structure_loss += step_structure_loss

        return structure_loss