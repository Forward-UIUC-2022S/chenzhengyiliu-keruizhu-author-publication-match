import torch
from sentence_transformers import SentenceTransformer
import networkx as nx
import pickle
from collections import defaultdict
from tqdm import tqdm
import random

co_author_graph:nx.Graph = nx.read_gpickle('co_author.gpickle')
with open('magfid2pids.pickle', 'rb') as f_in:
    magfid2pids = pickle.load(f_in)
with open('pid2infos.pickle', 'rb') as f_in:
    pid2infos = pickle.load(f_in)
sentence_transformer = SentenceTransformer('allenai-specter').eval().cuda()
    
interested_authors = [node for node in co_author_graph.nodes if len(magfid2pids[node]) >= 5]
sub_co_author_graph:nx.Graph = co_author_graph.subgraph(interested_authors)
node2emb = defaultdict(dict)
node_batch = []
sent_length_batch = []
sent_batch = []
for node in tqdm(sub_co_author_graph.nodes, total=sub_co_author_graph.number_of_nodes()):
    node_batch.append(node)
    known_pids = magfid2pids[node]
    random.seed(0)
    known_pubs = random.sample([pid2infos[pid] for pid in known_pids], min(10, len(known_pids)))
    sent_length_batch.append(len(known_pubs))
    sent_batch.extend(known_pubs)
    if len(node_batch) >= 100:
        known_emb = sentence_transformer.encode(['%s[SEP]%s' % (str(info.get('title')), str(info.get('abstract'))) for info in sent_batch], convert_to_tensor=True)
        for i, emb in enumerate(torch.split(known_emb, sent_length_batch)):
            node2emb[node_batch[i]]['emb'] = emb.mean(axis=0).cpu().numpy()
        node_batch = []
        sent_length_batch = []
        sent_batch = []
known_emb = sentence_transformer.encode(['%s[SEP]%s' % (str(info.get('title')), str(info.get('abstract'))) for info in sent_batch], convert_to_tensor=True)
for i, emb in enumerate(torch.split(known_emb, sent_length_batch)):
    node2emb[node_batch[i]]['emb'] = emb.mean(axis=0).cpu().numpy()
    
nx.set_node_attributes(sub_co_author_graph, node2emb)
with open('co_author_train.pickle', 'wb') as f_out:
    pickle.dump(sub_co_author_graph, f_out)