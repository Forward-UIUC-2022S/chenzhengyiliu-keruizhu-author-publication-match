from typing import List, Dict, Set
import networkx as nx
from collections import defaultdict
import mysql.connector
from itertools import combinations

def generate_graphs(core_author:str, candidate_ids:List[int], fid2pids:Dict[int, list], cursor:mysql.connector.cursor_cext.CMySQLCursor) -> List[nx.Graph]:
    temp_candidate_pids = []
    temp_candidate_split = []
    for candidate_id in candidate_ids:
        temp_pids = fid2pids[candidate_id]
        temp_candidate_pids.extend(temp_pids)
        temp_candidate_split.append(len(temp_pids))
        
    # Collect co_authors for this author
    pid2coauthor = get_magfid_from_mysql_by_pids(set(temp_candidate_pids), cursor)
    authors = set()
    for _, co_authors in pid2coauthor.items():
        authors.update(co_authors)
    magfid2pids = get_pid_from_mysql_by_magfids(authors, cursor)
    magfid2fname = get_fname_from_mysql_by_magfids(authors, cursor)
    pids = set()
    for _, pid in magfid2pids.items():
        pids.update(pid)
    pid2infos = get_pub_info_from_mysql(pids, cursor)
    start_idx = 0
    graphs = []
    core_author_ids = []
    for split in temp_candidate_split:
        graph = nx.Graph()
        core_author_id = None
        node_attributes = defaultdict(dict)
        for offset in range(split):
            temp_pid = temp_candidate_pids[start_idx + offset]
            co_authors = pid2coauthor[temp_pid]
            for co_author in co_authors:
                if co_author in node_attributes:
                    continue
                node_attributes[co_author]['pubs'] = [pid2infos[pid] for pid in magfid2pids[co_author]]
                node_attributes[co_author]['id'] = co_author
                if magfid2fname[co_author] == core_author:
                    core_author_id = co_author
            graph.add_edges_from(combinations(co_authors, 2))
        nx.set_node_attributes(graph, node_attributes)
        graphs.append(graph)
        core_author_ids.append(core_author_id)
        start_idx += split
    return graphs, core_author_ids

def get_pub_info_from_mysql(pids:Set[int], cursor:mysql.connector.cursor_cext.CMySQLCursor):
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

def get_magfid_from_mysql_by_pids(pids:Set[int], cursor:mysql.connector.cursor_cext.CMySQLCursor):
    cursor.execute('select PaperId, AuthorId from paperauthoraffiliations where PaperId in (%s)' % ','.join([str(pid) for pid in pids]))
    myresult = cursor.fetchall()
    paperid2authorids = defaultdict(list)
    for paperid, authorid in myresult:
        paperid2authorids[paperid].append(authorid)
    return paperid2authorids
    
def get_fname_from_mysql_by_magfids(fids:Set[int], cursor:mysql.connector.cursor_cext.CMySQLCursor):
    cursor.execute('select AuthorId, NormalizedName from authors where AuthorId in (%s)' % ','.join([str(fid) for fid in fids]))
    myresult = cursor.fetchall()
    authorid2fname = defaultdict(str)
    for authorid, fname in myresult:
        authorid2fname[authorid] = fname
    return authorid2fname

def get_pid_from_mysql_by_magfids(fids:Set[int], cursor:mysql.connector.cursor_cext.CMySQLCursor):
    cursor.execute('select AuthorId, PaperId from paperauthoraffiliations where AuthorId in (%s)' % ','.join([str(fid) for fid in fids]))
    myresult = cursor.fetchall()
    authorid2paperids = defaultdict(list)
    for authorid, paperid in myresult:
        authorid2paperids[authorid].append(paperid)
    return authorid2paperids