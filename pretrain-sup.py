import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from tqdm.auto import tqdm

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from ogb.nodeproppred import PygNodePropPredDataset

# custom modules
from maskgae.model import GNNEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="arxiv", help="dataset (default: arxiv)")
parser.add_argument("--root", type=str, default="data/", help="dataset directory")
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

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size for link prediction training. (default: 2**16)')

parser.add_argument("--start", type=str, default="node", help="Which Type to sample starting nodes for random walks, (default: node)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs. (default: 200)')
parser.add_argument("--save_path", required=True, type=str, help="path for the supervised pretrained model.")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()


args.bn = True
device = torch.device(f'cuda:{args.device}')
seed_everything(2024)

assert args.dataset in ['arxiv', 'mag']
print('loading ogb dataset...')
dataset = PygNodePropPredDataset(root=args.root, name=f'ogbn-{args.dataset}')
if args.dataset == 'mag':
    rel_data = dataset[0]
    # We are only interested in paper <-> paper relations.
    data = Data(
            x=rel_data.x_dict['paper'],
            edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
            y=rel_data.y_dict['paper'])
    data = T.ToUndirected()(data)
    split_idx = dataset.get_idx_split()
    data.train_mask = index_to_mask(split_idx['train']['paper'], data.num_nodes)
    data.val_mask = index_to_mask(split_idx['valid']['paper'], data.num_nodes)
    data.test_mask = index_to_mask(split_idx['test']['paper'], data.num_nodes)
else:
    data = T.ToUndirected()(dataset[0])
    split_idx = dataset.get_idx_split()
    data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
    data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)

data.y = data.y.squeeze()
data = data.to(device)

proj = nn.Sequential(nn.Linear(args.hidden_channels, dataset.num_classes))
model = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout, output_proj=proj,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

pbar = tqdm(range(1, args.epochs + 1))
best_val_acc = test_acc = 0
times = []
for epoch in pbar:
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    pbar.set_description(f'Train={train_acc:.2%}, Val={val_acc:.2%}, Test={test_acc:.2%}')


torch.cuda.empty_cache()

print(f'Saving supervised pretrianed model weights to {args.save_path}')
torch.save(model.state_dict(), args.save_path)