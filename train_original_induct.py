import numpy as np
import random
import time
import argparse
import time
import utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import torch_sparse
import os 
import gc
import math
import faiss

from utils import *
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.conv import MessagePassing
from sklearn.metrics import recall_score, precision_score
from models.basicgnn_large import GCN as GCN_PYG, GIN as GIN_PYG, SGC as SGC_PYG, GraphSAGE as SAGE_PYG, JKNet as JKNet_PYG
from models.mlp import MLP as MLP_PYG
from models.parametrized_adj_lp import PGE_Edge

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--parallel_gpu_ids', type=list, default=[0, 1, 2], help='gpu id')
parser.add_argument('--dataset', type=str, default='amazon-products')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--inference', type=bool, default=False)
#gnn
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--model', type=str, default='GCN')
#condensation
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--save', type=int, default=1)
#loop and validation
parser.add_argument('--teacher_model_loop', type=int, default=1000)

args = parser.parse_args()
print(args)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(args.gpu_id)
device='cuda'
print("Let's use", torch.cuda.device_count(), "GPUs!")

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def train_on_original_graph():
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=1e-5)

    best_val = 0
    best_test = 0
    start = time.perf_counter()
    for j in range(args.teacher_model_loop+1):
        model.train()
        optimizer.zero_grad()
        if args.model!='MLP':
            output_syn = model.forward(feat_train.to(device), adj_train.to(device))
        else:
            output_syn = model.forward(feat_train.to(device))
        loss = F.nll_loss(output_syn, labels_train)
        loss.backward()
        optimizer.step()

        if j % 100 == 0:
            if args.model!='MLP':
                output_train = model.predict(feat_train.to(device), adj_train.to(device))
                output_val = model.predict(feat_val.to(device), adj_val.to(device))
                output_test = model.predict(feat_test.to(device), adj_test.to(device))
            else:
                output_train = model.predict(feat_train.to(device))
                output_val = model.predict(feat_val.to(device))
                output_test = model.predict(feat_test.to(device))

            acc_train = utils.accuracy(output_train, labels_train)
            acc_val = utils.accuracy(output_val, labels_val)
            acc_test = utils.accuracy(output_test, labels_test)
            
            print(f'Epoch: {j:02d},'
                    f'Train: {100 * acc_train.item():.2f}%,'
                    f'Valid: {100 * acc_val.item():.2f}%,'
                    f'Test: {100 * acc_test.item():.2f}%')
            
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                if args.save:
                    torch.save(model.state_dict(), f'{root}/saved_model_large/teacher/{args.dataset}_{args.model}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt')
    end = time.perf_counter()
    print('Traing on the Original Graph Duration:', round(end-start), 's')
    print("Best Test Acc:", best_test)


if __name__ == '__main__':
    root=os.path.abspath(os.path.dirname(__file__))
    data = get_dataset(args.dataset, args.normalize_features)#get a Pyg2Dpr class, contains all index, adj, labels, features
    data = Transd2Ind(data, keep_ratio=args.keep_ratio)
    feat=torch.FloatTensor(data.features).detach().cuda()
    labels=torch.LongTensor(data.labels).cuda()
    idx_train, idx_val, idx_test=data.idx_train, data.idx_val, data.idx_test
    feat_train, feat_val, feat_test = feat[idx_train], feat[idx_val], feat[idx_test]
    adj_train, adj_val, adj_test=utils.to_tensor(data.adj_train, device='cpu'), utils.to_tensor(data.adj_val, device='cpu'), utils.to_tensor(data.adj_test, device='cpu')
    labels_train, labels_val, labels_test=labels[idx_train], labels[idx_val], labels[idx_test]
    d = feat.shape[1]
    nclass= int(labels.max()+1)

    if args.model=='GCN':
        model = GCN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).cuda()
    elif args.model=='SGC':
        model = SGC_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=0, nlayers=args.nlayers, sgc=True).cuda()
    elif args.model=='SAGE':
        model = SAGE_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).cuda()
    elif args.model=='GIN':
        model = GIN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).cuda()
    elif args.model=='JKNet':
        model = JKNet_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers+1, norm='BatchNorm', jk='cat', act=args.activation).cuda()
    else:
        model = MLP_PYG(channel_list=[d, args.hidden, args.hidden, nclass], dropout=[args.dropout, args.dropout, args.dropout], num_layers=3, norm='BatchNorm', act=args.activation).to(device)
    model.initialize()
    if args.model in ['GCN', 'SGC', 'JKNet']:
        if utils.is_sparse_tensor(adj_train):
            adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
            adj_val = utils.normalize_adj_tensor(adj_val, sparse=True)
            adj_test = utils.normalize_adj_tensor(adj_test, sparse=True)
        else:
            adj_train = utils.normalize_adj_tensor(adj_train)
            adj_val = utils.normalize_adj_tensor(adj_val)
            adj_test = utils.normalize_adj_tensor(adj_test)
        adj_train = SparseTensor(row=adj_train._indices()[0], col=adj_train._indices()[1],value=adj_train._values(), sparse_sizes=adj_train.size()).t()
        adj_val = SparseTensor(row=adj_val._indices()[0], col=adj_val._indices()[1],value=adj_val._values(), sparse_sizes=adj_val.size()).t()
        adj_test = SparseTensor(row=adj_test._indices()[0], col=adj_test._indices()[1],value=adj_test._values(), sparse_sizes=adj_test.size()).t()
    else:#add self loops
        adj_train = SparseTensor(row=torch.concat((adj_train._indices()[0], torch.arange(len(labels_train)))), col=torch.concat((adj_train._indices()[1],torch.arange(len(labels_train)))), value=torch.concat((adj_train._values(),torch.ones((len(labels_train),)))), sparse_sizes=adj_train.size()).t()
        adj_val = SparseTensor(row=torch.concat((adj_val._indices()[0], torch.arange(len(labels_val)))), col=torch.concat((adj_val._indices()[1],torch.arange(len(labels_val)))), value=torch.concat((adj_val._values(),torch.ones((len(labels_val),)))), sparse_sizes=adj_val.size()).t()
        adj_test = SparseTensor(row=torch.concat((adj_test._indices()[0], torch.arange(len(labels_test)))), col=torch.concat((adj_test._indices()[1],torch.arange(len(labels_test)))), value=torch.concat((adj_test._values(),torch.ones((len(labels_test),)))), sparse_sizes=adj_test.size()).t()

    train_on_original_graph()
