import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import dgl
import torch

from dgllife.data import MoleculeCSVDataset 

class EnzymesDataset(MoleculeCSVDataset):
    def __init__(self, smiles, graphs, labels, masks):
        self.smiles = smiles
        self.graphs = graphs
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.graphs[idx], self.labels[idx], self.masks[idx]



def read_graphfile(datadir, dataname, max_nodes=None):
    """Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = os.path.join(datadir, dataname)
    filename_graph_indic = prefix + "_graph_indicator.txt"
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + "_node_labels.txt"
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print("No node labels")

    filename_node_attrs = prefix + "_node_attributes.txt"
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [
                    float(attr) for attr in re.split("[,\s]+", line) if not attr == ""
                ]
                node_attrs.append(np.array(attrs))
    except IOError:
        print("No node attributes")

    filename_graphs = prefix + "_graph_labels.txt"
    graph_labels = []

    # assume that all graph labels appear in the dataset
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    graph_labels = np.array(graph_labels)

    filename_adj = prefix + "_A.txt"
    base_idx = min(graph_indic.values()) - 1
    adj_list = {i + base_idx: [] for i in range(1, len(graph_labels) + 1)}

    base_e = 1e7
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            base_e = min(base_e, min(e0, e1))
    base_e -= 1

    new_graph_indic = {}
    for k, v in graph_indic.items():
        new_graph_indic[k + base_e] = v
    graph_indic = new_graph_indic

    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[base_idx + i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph["label"] = graph_labels[i - 1]
        for u in G.nodes:
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1 - base_e]
                node_label_one_hot[node_label] = 1
                G.nodes[u]["label"] = node_label_one_hot
            if len(node_attrs) > 0:
                G.nodes[u]["h"] = node_attrs[u - 1 - base_e]
        if len(node_attrs) > 0:
            G.graph["h_dim"] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        for n in G.nodes:
            mapping[n] = it
            it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))

    # use node labels as feature
    for G in graphs:
        for u in G.nodes:
            G.nodes[u]["h"] = np.array(G.nodes[u]["label"], dtype=np.float32)

    return graphs

def convert_to_dglgraph(nx_graphs):
    dgl_graphs = []
    for nx_graph in nx_graphs:
        # TODO: change this one since we want to keep the graph label
        dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['h'])
        dgl_graphs.append(dgl_graph)
    
    return dgl_graphs

def create_dgl_dataset(datadir, dataname, max_nodes=None, device='cpu'):
    nx_graphs = read_graphfile(datadir, dataname, max_nodes=max_nodes)
    dgl_graphs = convert_to_dglgraph(nx_graphs)
    n = len(dgl_graphs)
    smiles = [""] * n
    labels, masks = [], []
    for g in nx_graphs:
        if 'label' in g.graph:
            labels.append(torch.tensor([g.graph['label']], dtype=torch.float32, device=device))
            masks.append(torch.ones(1, dtype=torch.float32, device=device))
        else:
            labels.append(torch.zeros(1, dtype=torch.float32, device=device))
            masks.append(torch.zeros(1, dtype=torch.float32, device=device))


    labels = torch.cat(labels)
    masks = torch.cat(masks)
    dataset = EnzymesDataset(smiles, dgl_graphs, labels, masks)

    return dataset

def read_classes(datadir):
    task_dirs = sorted([p for p in Path(datadir).glob("task*")])
    cls_per_task = []
    for p in task_dirs:
        classes = json.load(open(p / 'classes.json', 'r'))['new_classes']
        classes = [int(c) for c in classes]
        cls_per_task.append(classes)
    
    return cls_per_task

