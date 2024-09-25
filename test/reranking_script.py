import numpy as np
import torch

from ..reranking import sgr, gnn, krnn
from ..evaluators.retrieval import Retrieval
from ..utils.utils import load_config

features_load_path = 'cluster/rl-map/dataset/pfs_tf_smppthAP_256.npz'
loaded_features = np.load(features_load_path)

pfs_tf = loaded_features['pfs_tf']
writer = loaded_features['writer']

_eval = Retrieval()
pfs_tf_rerank = sgr.sgr_reranking(torch.tensor(pfs_tf), k=2, layer=1, gamma=0.4)[0]

breakpoint()

res, _ = _eval.eval(pfs_tf, writer)
res_rerank, _ = _eval.eval(pfs_tf_rerank, writer)

print(f'''MAP: {res['map']}''')
print(f'''Top-1: {res['top1']}''')
print(f'''rerank MAP: {res_rerank['map']}''')
print(f'''rerank Top-1: {res_rerank['top1']}''')

#if __name__ = '__main__':

#    parser = argparse.ArgumentParser()

#    parser.add_argument('--path', default ='cluster/rl-map/dataset/pfs_tf_smppthAP_256.npz')

#    args = parser.parse_args()
#
config = load_config(args)[0]
