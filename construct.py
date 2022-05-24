import sys
from pyHGT.data import *
import networkx as nx
import pickle

graph = renamed_load(open('/home/cl115/mag/org_OAG/data_graph/graph_CS.pk', 'rb'))
G = nx.Graph()
timeline = 2016


author_networkx = []
for i, j in tqdm(graph.node_feature['author'].iterrows(), total = len(graph.node_feature['author'])):
    author_networkx.append((i, {"name": j["name"], "embedding": j["emb"]}))
G.add_nodes_from(author_networkx)

for i, j in tqdm(graph.node_feature['paper'].iterrows(), total = len(graph.node_feature['paper'])):
    p_id = i
    for _type in graph.edge_list['paper']['author']:
        if p_id in graph.edge_list['paper']['author'][_type]:
            author_list = []
            for a_id in graph.edge_list['paper']['author'][_type][p_id]:
                _time = graph.edge_list['paper']['author'][_type][p_id][a_id]
                if _time <= 2016:
                    author_list.append(a_id)
                else:
                    break
            for i1 in range(len(author_list) - 1):
                for i2 in range(i1 + 1, len(author_list)):
                    G.add_edge(author_list[i1], author_list[i2])

with open("data/co_author_graph.pickle", "wb") as f:
    pickle.dump(G, f)


name_to_idx = {}
idx_to_name = {}
for i, j in tqdm(graph.node_feature['author'].iterrows(), total = len(graph.node_feature['author'])):
    if j['name'] not in name_to_idx:
        name_to_idx[j['name']] = []
    name_to_idx[j['name']] += [i]
    idx_to_name[i] = j['name']

with open("data/name_to_idx.pickle", 'wb') as f:
    pickle.dump(name_to_idx, f)
with open("data/idx_to_name.pickle", 'wb') as f:
    pickle.dump(idx_to_name, f)


# for ranking.
train_list = []
for i, j in tqdm(graph.node_feature['paper'].iterrows(), total = len(graph.node_feature['paper'])):
    p_id = i
    for _type in graph.edge_list['paper']['author']:
        if p_id in graph.edge_list['paper']['author'][_type]:
            author_list = []
            for a_id in graph.edge_list['paper']['author'][_type][p_id]:
                _time = graph.edge_list['paper']['author'][_type][p_id][a_id]
                if _time > 2016:
                    author_list.append(a_id)
                else:
                    break
            for a_id in author_list:
                a_name = idx_to_name[a_id]
                new_list = author_list.copy()
                new_list.remove(a_id)
                if len(name_to_idx[a_name]) >= 4 and len(new_list) >= 2:
                    train_list.append([a_id, name_to_idx[a_name], new_list])

print(len(train_list))
with open("data/train_list.pickle", "wb") as f:
    pickle.dump(train_list, f)


