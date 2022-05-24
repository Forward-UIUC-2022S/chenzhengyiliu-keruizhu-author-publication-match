from typing import List, Dict
import networkx as nx
from collections import defaultdict
import mysql.connector

def generate_graphs(candidate_ids:List[int], fid2pids:Dict[int, list], pid2fids:Dict[int, list]) -> List[nx.Graph]:
    candidate_pid_sets = [fid2pids[candidate_id] for candidate_id in candidate_ids]
    neighbor_sets = []
    for i, pid_set in enumerate(candidate_pid_sets):
        neighbors = set()
        for pid in pid_set:
            neighbors.update(pid2fids[pid])
        neighbors.remove(candidate_ids[i])
        neighbor_sets.append(neighbors)
    graphs = []
    for i in range(len(candidate_ids)):
        g = nx.Graph()
        g.add_node(candidate_ids[i])
        g.add_edges_from([(neighbor, candidate_ids[i]) for neighbor in neighbor_sets[i]])
        node_attributes = {node : {'pids' : fid2pids[node]} for node in g.nodes}
        nx.set_node_attributes(g, node_attributes)
        graphs.append(g)
    return graphs

def get_pub_info_from_mysql(pids:List[int], cursor:mysql.connector.cursor_cext.CMySQLCursor):
    pub_info_dict = defaultdict(dict)
    cursor.execute('select PaperId, PaperTitle from papers where PaperId in (%s)' % ','.join([str(pid) for pid in pids]))
    myresult = cursor.fetchall()
    for x in myresult:
        pub_info_dict[x[0]]['title'] = x[1]
    cursor.execute('select PaperId, Abstract from paperabstracts where PaperId in (%s)' % ','.join([str(pid) for pid in pids]))
    myresult = cursor.fetchall()
    for x in myresult:
        pub_info_dict[x[0]]['abstract'] = x[1]
    return pub_info_dict