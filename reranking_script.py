import numpy as np
import torch

from reranking import sgr, gnn, krnn
from evaluators.retrieval import Retrieval
from utils.utils import load_config


"""
Reranking using the provided rerank.py, when provided with the features. 
Prints the mAP and Top-1 values for each of the reanking methods.
"""

features_load_path = '/cluster/qy41tewa/rl-map/experiments/triplet_loss_model_reranking-2024-07-15-14-40/pfs_tf_triplet_1024_16.npz'
loaded_features = np.load(features_load_path)

pfs_tf = loaded_features['pfs_tf']
writer = loaded_features['writer']

_eval = Retrieval()
rerank_sgr = sgr.sgr_reranking(torch.tensor(pfs_tf), k=1, layer=1, gamma=0.4)[0]
rerank_gnn = gnn.gnn_reranking(torch.tensor(pfs_tf), k1=1,k2=2, layer=2)[0]
rerank_krnn = krnn.kRNN(torch.tensor(pfs_tf), k=3)

reranked_pfs_tf = [rerank_sgr, rerank_gnn, rerank_krnn]
rerank_names = ['SGR', 'GNN', 'KrNN']

#breakpoint()

res, _ = _eval.eval(pfs_tf, writer)
reranked_res = {}

for pfs, name in zip(reranked_pfs_tf, rerank_names):
    reranked_res[f'{name}'], _ = _eval.eval(pfs, writer)
    

print(f'''MAP: {res['map']}''')
print(f'''Top-1: {res['top1']}''')

for reranker, reranked_res in reranked_res.items():
    print(f'''{reranker} MAP: {reranked_res['map']}''')
    print(f'''{reranker} Top-1: {reranked_res['top1']}''')


#if __name__ = '__main__':

#    parser = argparse.ArgumentParser()

#    parser.add_argument('--path', default ='cluster/rl-map/dataset/pfs_tf_smppthAP_256.npz')

#    args = parser.parse_args()
#
#config = load_config(args)[0]
