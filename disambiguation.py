import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import pickle
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score

from tqdm import tqdm


torch.manual_seed(42)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(600, 628)  # 384 for the embedding dim of text 
        self.conv2 = GCNConv(628, 64)

    def encode(self, x, pos_edge_index):
        x = self.conv1(x, pos_edge_index)
        x = x.relu()
        return self.conv2(x, pos_edge_index)

    def decode(self, z, edge_index):
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z): 
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list 


def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(graph, model, optimizer):
    model.train()

    train_loader = NeighborLoader(data=graph, num_neighbors=[5, 10, 15], batch_size=128, 
                               shuffle=True, num_workers=12, directed = False)

    for sampled_data in train_loader:
      pos_edge_index = sampled_data.edge_index
      neg_edge_index = negative_sampling(edge_index=pos_edge_index,
        num_nodes=sampled_data.num_nodes, num_neg_samples=pos_edge_index.size(1))

      optimizer.zero_grad()
    
      z = model.encode(sampled_data.x.float(), pos_edge_index)
      edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
      link_logits = model.decode(z, edge_index)
      link_labels = get_link_labels(pos_edge_index, neg_edge_index)
      loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
      loss.backward()
      optimizer.step()


@torch.no_grad()
def validation(data):
    model.eval()
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(edge_index=pos_edge_index,
        num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))

    z = model.encode(data.x.float(), data.edge_index)
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    out = model.decode(z, edge_index).view(-1).sigmoid()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    return roc_auc_score(link_labels.cpu().numpy(), out.cpu().numpy())


@torch.no_grad()
def scoring_one_author(z, test_pair, name_to_idx):
    model.eval()
    
    candidate_id = test_pair[0]
    candidate_name = test_pair[1]
    references_name = test_pair[2]
    z_c = z[candidate_id]

    res = []
    for i in references_name:
        references_id = np.array(name_to_idx[i])
        z_r = z[references_id]
        pred = torch.max(z_c @ z_r.T).sigmoid()
        res.append(pred.cpu())

    return np.mean(res)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("data/co_author_graph.pickle", "rb") as f:
        G = pickle.load(f)
    
    with open("data/name_to_idx.pickle", 'rb') as f:
        name_to_idx = pickle.load(f)

    graph = from_networkx(G, group_node_attrs=['embedding'])

    model, graph = Net().to(device), graph.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    
    Train = True
    if Train:
        val_loader = NeighborLoader(data=graph, num_neighbors=[15, 10, 5], batch_size=524, 
                               shuffle=True, num_workers=12, directed = False)
        it = iter(val_loader) 

        best_score = 0
        for epoch in range(10):
            train(graph, model, optimizer)
            sampled_data = next(it)
            acc = validation(sampled_data)
            print(f"test auc score: {acc: .4f}")
            if acc > best_score:
                best_score = acc
                torch.save(model.state_dict(), "save_model/checkpoints.gcn")

    model.load_state_dict(torch.load("save_model/checkpoints.gcn"))
    model.eval()
    # embedding all nodes on the graph
    embeddings = model.encode(graph.x.float(), graph.edge_index)

    with open("data/test_list.pickle", "rb") as f:
        test_pairs = pickle.load(f)

    res = scoring_one_author(embeddings, test_pairs[0], name_to_idx)

    print(f"test co-author prediction score: {res: .4f}")
