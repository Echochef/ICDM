import argparse
import pandas as pd
import os.path as osp

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, scatter

# custom modules
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from maskgae.mask import MaskEdge, MaskPath


def train_linkpred(model, data, args, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    for epoch in tqdm(range(1, 1 + args.epochs)):
        model.train()
        loss = model.train_epoch(data.to(device), optimizer,
                                 alpha=args.alpha, 
                                 batch_size=args.batch_size)


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="data/icdm2023_session1_test", help="path to the data directory. ")
parser.add_argument("--layer", type=str, default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", type=str, default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder layers. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 128)')
parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.2, help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--decoder_dropout', type=float, default=0.0, help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate for training. (default: 0.0005)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size for link prediction training. (default: 2**16)')

parser.add_argument("--start", type=str, default="node", help="Which Type to sample starting nodes for random walks, (default: node)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument('--epochs', type=int, default=5, help='Number of fine-tuning epochs. (default: 5)')
parser.add_argument("--pretrain_path", required=True, type=str, 
                    help="path for the pretrained model")
parser.add_argument("--embedding_save_path", required=True, type=str, 
                    help="save path for the pretrained embeddings")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()


args.bn = True
device = torch.device(f'cuda:{args.device}')
seed_everything(2024)

print('Loading Data...')
edge_index = pd.read_csv(f'{args.root}/icdm2023_session1_test_edge.txt', header=None)
x = pd.read_csv(f'{args.root}/icdm2023_session1_test_node_feat.txt', header=None)

data = Data(edge_index=torch.from_numpy(edge_index.to_numpy().T).long(),
            x=torch.from_numpy(x.to_numpy().astype('float32')))

device = torch.device(f'cuda:{args.device}')
data = data.to(device)

mask = MaskEdge(p=args.p)
proj = nn.Sequential(nn.BatchNorm1d(100), nn.Linear(100, 128), nn.ReLU())
encoder = GNNEncoder(128, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout, input_proj=proj,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

print(f'Loading pretrianed model weights from {args.pretrain_path}')
encoder.load_state_dict(torch.load(args.pretrain_path), strict=False)

edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                           num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

print('Link Prediction Fine-Tuning...')
train_linkpred(model, data, args, device=device)


data.edge_index = to_undirected(data.edge_index)
embedding = model.encoder.get_embedding(data.x, data.edge_index).cpu()


print('Neighborhood Aggregation...')
edge_index = add_self_loops(data.edge_index)[0]
row, col = edge_index.cpu()
x = embedding
hops = 2
for _ in tqdm(range(hops)):
    x = scatter(x[col], row, dim=0, 
                dim_size=x.size(0), 
                reduce='sum')
x = torch.cat([x, embedding], dim=1)

torch.save(x.contiguous(), args.embedding_save_path)