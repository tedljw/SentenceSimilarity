from bert_serving.client import BertClient
import numpy as np
bc = BertClient(ip="127.0.0.1")

def cosine(a,b):
    return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

emb=np.array(bc.encode(['First do it', 'then do it right']))

print(['First do it', 'then do it right'],":",cosine(emb[0],emb[1]))
