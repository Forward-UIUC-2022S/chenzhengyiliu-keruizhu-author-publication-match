from statistics import mean
from sentence_transformers import SentenceTransformer, util
import pickle
from flask import Flask, jsonify, request
import traceback
import mysql.connector
from cs411_util import generate_graphs, get_pub_info_from_mysql
from typing import List, Dict
import networkx as nx
from collections import defaultdict
from itertools import combinations
from statistics import mean
from disambiguation import Net
import torch
from torch_geometric.utils import from_networkx

app = Flask(__name__)

with open('awfid2pids.pickle', 'rb') as f_in:
    awfid2pids = pickle.load(f_in)
with open('pid2awfids.pickle', 'rb') as f_in:
    pid2awfids = pickle.load(f_in)
with open('fname2awfids.pickle', 'rb') as f_in:
    fname2awfids = pickle.load(f_in)
with open('magfid2pids.pickle', 'rb') as f_in:
    magfid2pids = pickle.load(f_in)
with open('pid2magfids.pickle', 'rb') as f_in:
    pid2magfids = pickle.load(f_in)
with open('fname2magfids.pickle', 'rb') as f_in:
    fname2magfids = pickle.load(f_in)
with open('awfid2fname.pickle', 'rb') as f_in:
    awfid2fname = pickle.load(f_in)
with open('pid2infos.pickle', 'rb') as f_in:
    pid2infos = pickle.load(f_in)
with open('magfid2fname.pickle', 'rb') as f_in:
    magfid2fname = pickle.load(f_in)
    
sentence_transformer = SentenceTransformer('allenai-specter').eval()

with open('co_author_train.pickle', 'rb') as f_in:
    sub_co_author_graph:nx.Graph = pickle.load(f_in)
id2idx = {id : idx for idx, id in enumerate(sub_co_author_graph.nodes)}

db = mysql.connector.connect(user='mag_readonly', password='j6gi48ch82nd9pff', host="mag-2020-09-14.mysql.database.azure.com",
   port=3306,
   database='mag_2020_09_14',
   ssl_ca="DigiCertGlobalRootCA.crt.pem",
   ssl_disabled=False)
cursor = db.cursor()

nx_model = Net()
nx_model.load_state_dict(torch.load("save_model/checkpoints.gcn"))
nx_model.eval()
# embedding all nodes on the graph
graph = from_networkx(sub_co_author_graph, group_node_attrs=['emb'])
embeddings = nx_model.encode(graph.x.float(), graph.edge_index)

# def generate_graphs_fast(core_author:str, candidate_ids:List[int], fid2pids:Dict[int, list]) -> List[nx.Graph]:
#     temp_candidate_pids = []
#     temp_candidate_split = []
#     for candidate_id in candidate_ids:
#         temp_pids = fid2pids[candidate_id]
#         temp_candidate_pids.extend(temp_pids)
#         temp_candidate_split.append(len(temp_pids))
        
#     # Collect co_authors for this author
#     pid2coauthor = {pid : pid2magfids[pid] for pid in temp_candidate_pids}
#     authors = set()
#     for _, co_authors in pid2coauthor.items():
#         authors.update(co_authors)
    
#     temp_magfid2pids = {fid : magfid2pids[fid] for fid in authors}
#     pids = set()
#     for _, pid in temp_magfid2pids.items():
#         pids.update(pid)
#     temp_pid2infos = {pid : pid2infos[pid] for pid in pids}
#     start_idx = 0
#     graphs = []
#     core_author_ids = []
#     for split in temp_candidate_split:
#         graph = nx.Graph()
#         core_author_id = None
#         node_attributes = defaultdict(dict)
#         for offset in range(split):
#             temp_pid = temp_candidate_pids[start_idx + offset]
#             co_authors = pid2coauthor[temp_pid]
#             for co_author in co_authors:
#                 if co_author in node_attributes:
#                     continue
#                 node_attributes[co_author]['pubs'] = [temp_pid2infos[pid] for pid in temp_magfid2pids[co_author]]
#                 node_attributes[co_author]['id'] = co_author
#                 if temp_magfid2pids[co_author] == core_author:
#                     core_author_id = co_author
#             graph.add_edges_from(combinations(co_authors, 2))
#         nx.set_node_attributes(graph, node_attributes)
#         graphs.append(graph)
#         core_author_ids.append(core_author_id)
#         start_idx += split
#     return graphs, core_author_ids

print('Resource loaded')

@app.route('/', methods=['GET'])
def index():
    title = request.args.get('title')
    abstract = request.args.get('abstract')
    coauthors = request.args.get('coauthor')
    target = request.args.get('target')

    # sanity check
    if target is None:
        return jsonify({'error' : 'target author is not provided'})
    if (title is None or abstract is None) and coauthors is None:
        return jsonify({'error' : 'information is not complete'})
    try:
        target_author_id = int(target)
    except:
        return jsonify({'error' : 'target author should be an integer'})
    if target_author_id not in awfid2pids:
        return jsonify({'error' : 'target author is not in the academicworld database'})
    
    semantic_score = 0
    if title is not None and abstract is not None:
        try:
            known_pids = awfid2pids[target_author_id]
            known_pubs = get_pub_info_from_mysql(known_pids, cursor)
            known_emb = sentence_transformer.encode(['%s[SEP]%s' % (str(info.get('title')), str(info.get('abstract'))) for pid, info in known_pubs.items()], convert_to_tensor=True).cuda()
            unknown_emb = sentence_transformer.encode(['%s[SEP]%s' % (title, abstract)], convert_to_tensor=True).cuda()
            search_hits = util.semantic_search(unknown_emb, known_emb, top_k=1)
            semantic_score = search_hits[0][0]['score']
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            pass
    
    co_author_score = None
    if coauthors is not None:
        print(1)
        coauthors = eval(coauthors)
        target_author_name = awfid2fname[target_author_id]
        if target_author_name in fname2magfids:
            # Locate target author in MAG
            target_author_pids = awfid2pids[target_author_id]
            temp_coauthor_for_target_author = [set(pid2magfids[pid]) for pid in target_author_pids if pid in pid2magfids]
            if temp_coauthor_for_target_author:
                target_author_magfid = None
                for fids in temp_coauthor_for_target_author:
                    if not fids:
                        continue
                    for fid in fids:
                        if magfid2fname[fid] == target_author_name:
                            target_author_magfid = fid
                            break
                    if target_author_magfid:
                        break
                if target_author_magfid in id2idx:
                    target_author_emb = embeddings[id2idx[target_author_magfid]]
                    # Filter out coauthors not in the graph
                    recognized_coauthors = [fname2magfids[a] for a in coauthors if a in fname2magfids]
                    trained_coauthors = [[id for id in coauthor_ids if sub_co_author_graph.has_node(id)] for coauthor_ids in recognized_coauthors]
                    trained_coauthors = [coauthor_ids for coauthor_ids in trained_coauthors if coauthor_ids]
                    if trained_coauthors:
                        scores = []
                        for trained_coauthor in trained_coauthors:
                            coauthor_emb = torch.index_select(embeddings, 0, torch.tensor([id2idx[id] for id in trained_coauthor]))
                            temp_score = torch.max(target_author_emb @ coauthor_emb.T).sigmoid().cpu().item()
                            scores.append(temp_score)
                        co_author_score = mean(scores)

        # for coauthor in coauthors:
        #     if coauthor in fname2magfids:
        #         generate_graphs_fast(coauthor, fname2magfids[coauthor], magfid2pids)
    
    return jsonify({'semantic_score': semantic_score,
                    'semantic_conclusion' : semantic_score > 0.73,
                    'co_author_score' : co_author_score,
                    'co_author_conclusion' : co_author_score is not None and co_author_score > 0.65})

app.run(host="0.0.0.0")