from sentence_transformers import SentenceTransformer, util
import pickle
from flask import Flask, jsonify, request
import traceback
import mysql.connector
from cs411_util import generate_graphs, get_pub_info_from_mysql

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
    
sentence_transformer = SentenceTransformer('allenai-specter').eval()

db = mysql.connector.connect(user='mag_readonly', password='j6gi48ch82nd9pff', host="mag-2020-09-14.mysql.database.azure.com",
   port=3306,
   database='mag_2020_09_14',
   ssl_ca="DigiCertGlobalRootCA.crt.pem",
   ssl_disabled=False)
cursor = db.cursor()

print('Resource loaded')

@app.route('/', methods=['GET'])
def index():
    title = request.args.get('title')
    abstract = request.args.get('abstract')
    coauthor = request.args.get('coauthor')
    target = request.args.get('target')
    # sanity check
    if target is None:
        return jsonify({'error' : 'target author is not provided'})
    if (title is None or abstract is None) and coauthor is None:
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
    
    return jsonify({'score': semantic_score,
                    'semantic_score': semantic_score,
                    'belongs' : semantic_score > 0.73})

app.run(host="0.0.0.0")