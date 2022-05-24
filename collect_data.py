import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import json
import pickle

data_dir = 'MAG_0919_CS/'

print('Collect paper info')
Papers = data_dir + 'Papers_CS_20190919.tsv'
papers_df = pd.read_csv(Papers, sep='\t')
pid2infos = {item['PaperId'] : {'title' : item['NormalizedTitle'], 'year' : item['PublishYear']} for item in tqdm(papers_df[['PaperId', 'NormalizedTitle', 'PublishYear']].to_dict('records'))}

print('Collect abstract')
PAb = data_dir + 'PAb_CS_20190919.tsv'
pab_df = pd.read_csv(PAb, sep='\t')
pid2abs = {item['PaperId'] : item['Abstract'] for item in tqdm(pab_df.to_dict('records'))}

print('Collect paper-author info')
PAuAf = data_dir + 'PAuAf_CS_20190919.tsv'
pauaf_df = pd.read_csv(PAuAf, sep='\t')
pid2magfids = defaultdict(list)
magfid2pids = defaultdict(list)
for item in tqdm(pauaf_df[['PaperSeqid', 'AuthorSeqid']].to_dict('records')):
    pid2magfids[item['PaperSeqid']].append(item['AuthorSeqid'])
    magfid2pids[item['AuthorSeqid']].append(item['PaperSeqid'])

print('Collect name-to-author-id info')
fname2magfids = defaultdict(list)
magfid2fname = {}
for item in tqdm(pd.read_csv('SeqName_CS_20190919.tsv', sep='\t', header=None).to_dict('records')):
    if item[2] == 'author':
        fname2magfids[item[1].lower()].append(item[0])
        magfid2fname[item[0]] = item[1].lower()

print('Collect co-author graph')
co_author_graph = nx.Graph()
for pid, info in tqdm(pid2infos.items()):
    info['abstract'] = pid2abs.get(pid)
    co_authors = pid2magfids.get(pid)
    info['authors'] = co_authors
    for i in range(len(co_authors)):
        for j in range(i+1, len(co_authors)):
            if not co_author_graph.has_edge(co_authors[i], co_authors[j]):
                co_author_graph.add_edge(co_authors[i], co_authors[j], c=0)
            co_author_graph.get_edge_data(co_authors[i], co_authors[j])['c'] += 1
            
print('Save mag info')
with open('magfid2pids.pickle', 'wb') as f_out:
    pickle.dump(magfid2pids, f_out)
with open('pid2magfids.pickle', 'wb') as f_out:
    pickle.dump(pid2magfids, f_out)
with open('fname2magfids.pickle', 'wb') as f_out:
    pickle.dump(fname2magfids, f_out)
with open('pid2infos.pickle', 'wb') as f_out:
    pickle.dump(pid2infos, f_out)
with open('magfid2fname.pickle', 'wb') as f_out:
    pickle.dump(magfid2fname, f_out)
nx.write_gpickle(co_author_graph, 'co_author.gpickle')
print('done')

print('Collect academicworld data')
with open('faculty.json') as f_in:
    faculty = json.load(f_in)
with open('publications.json') as f_in:
    publications = json.load(f_in)
    
awfid2pids = {item['id'] : item['publications'] for item in faculty}
fname2awfids = defaultdict(list)
awfid2fname = {}
for item in faculty:
    fname2awfids[item['name'].lower()].append(item['id'])
    awfid2fname[item['id']] = item['name'].lower()
pid2awfids = defaultdict(list)
for fid, pids in awfid2pids.items():
    for pid in pids:
        pid2awfids[pid].append(fid)
        
with open('awfid2pids.pickle', 'wb') as f_out:
    pickle.dump(awfid2pids, f_out)
with open('pid2awfids.pickle', 'wb') as f_out:
    pickle.dump(pid2awfids, f_out)
with open('fname2awfids.pickle', 'wb') as f_out:
    pickle.dump(fname2awfids, f_out)
with open('awfid2fname.pickle', 'wb') as f_out:
    pickle.dump(awfid2fname, f_out)