import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from fast_pytorch_kmeans import KMeans

def reorder(x):
    """Reorder clusters, e.g.,
    2, 1, 0, 0 -> 0, 1, 2, 2
    """
    d = dict()
    i = 0
    out = []
    for c in x:
        if c not in d:
            d[c] = i
            i += 1
        out.append(d[c])
    return out

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_path", type=str, default="embedding_arxiv.pt", 
                    help="Save path for the pretrained embeddings. (default: embedding_arxiv.pt)")
parser.add_argument("--output", type=str, default="submit.txt", 
                    help="Output path for cluster results. (default: submit.txt)")
parser.add_argument('--k', type=int, default=15, help='Numbers of clusters.')
parser.add_argument('--seed', type=int, default=-1, help='seed for clustering.')
parser.add_argument('--runs', type=int, default=5, help='Number of trials for ensembling.')

args = parser.parse_args()



k = args.k
seed = args.seed
runs = args.runs
print(f'k={k}, seed={seed}, ensembled over {runs} runs')

print(f'Loading embedding from {args.embedding_path}')
x = torch.load(args.embedding_path)

print('Start ensembling...')
predicts = np.zeros((x.shape[0], k))
if seed >= 0:
    torch.manual_seed(seed)
# ensemble
for _ in range(runs):
    kmeans = KMeans(n_clusters=k, mode="cosine", init_method='kmeans++', verbose=1)
    clusters = kmeans.fit_predict(x)
    for i, c in tqdm(enumerate(reorder(clusters.tolist()))):
        predicts[i, c] += 1
        
predicts = predicts.argmax(1)
print(f'Results saved at {args.output}')
pd.DataFrame(predicts.tolist()).to_csv(args.output, index=None, header=None)