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

from utils import *
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.utils import coalesce

from models.basicgnn_large import GCN as GCN_PYG, GIN as GIN_PYG, SGC as SGC_PYG, GraphSAGE as SAGE_PYG, JKNet as JKNet_PYG
from models.mlp import MLP as MLP_PYG
from models.parametrized_adj import PGE


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--parallel_gpu_ids', type=list, default=[0,1], help='gpu id')
parser.add_argument('--dataset', type=str, default='reddit')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
#gnn
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--inference', type=bool, default=False)
parser.add_argument('--teacher_model', type=str, default='GCN')
parser.add_argument('--lr_teacher_model', type=float, default=0.002)
parser.add_argument('--save', type=int, default=1)
#loop and validation
parser.add_argument('--teacher_model_loop', type=int, default=600)
parser.add_argument('--teacher_val_stage', type=int, default=10)
args = parser.parse_args()
print(args)

device='cuda'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.cuda.set_device(args.gpu_id)
print("Let's use", torch.cuda.device_count(), "GPUs!")

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def train_teacher():
    start = time.perf_counter() 
    optimizer_origin=torch.optim.Adam(teacher_model.parameters(), lr=args.lr_teacher_model)

    best_val=0
    best_test=0
    for it in range(args.teacher_model_loop+1):
        #whole graph
        teacher_model.train()
        optimizer_origin.zero_grad()
        output = teacher_model.forward(feat_train.to(device), adj_train.to(device))
        loss = F.nll_loss(output, labels_train)
        loss.backward()
        optimizer_origin.step()

        if(it%args.teacher_val_stage==0):
            acc_train = utils.accuracy(output, labels_train)
            output = teacher_model.predict(feat_val.to(device), adj_val.to(device))
            acc_val = utils.accuracy(output, labels_val)
            output = teacher_model.predict(feat_test.to(device), adj_test.to(device))
            acc_test = utils.accuracy(output, labels_test)

            print(f'Epoch: {it:02d}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Train: {100 * acc_train.item():.2f}% '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
        
    end = time.perf_counter()
    print("Best Test:", best_test)
    print('Traing on the Original Graph Duration:', round(end-start), 's')
    return


if __name__ == '__main__':
    root=os.path.abspath(os.path.dirname(__file__))
    data = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data, keep_ratio=1)

    feat_train=torch.FloatTensor(data.feat_train).to('cpu')
    feat_val=torch.FloatTensor(data.feat_val).to('cpu')
    feat_test=torch.FloatTensor(data.feat_test).to('cpu')
    adj_train=utils.to_tensor(data.adj_train, device='cpu')
    adj_val=utils.to_tensor(data.adj_val, device='cpu')
    adj_test=utils.to_tensor(data.adj_test, device='cpu')
    labels=torch.LongTensor(data.labels).to(device)
    labels_train=torch.LongTensor(data.labels_train).to(device)
    labels_val=torch.LongTensor(data.labels_val).to(device)
    labels_test=torch.LongTensor(data.labels_test).to(device)

    d = feat_train.shape[1]
    nclass= int(labels.max()+1)
    del data
    gc.collect()

    if utils.is_sparse_tensor(adj_train):
        adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
        adj_val = utils.normalize_adj_tensor(adj_val, sparse=True)
        adj_test = utils.normalize_adj_tensor(adj_test, sparse=True)
    else:
        adj_train = utils.normalize_adj_tensor(adj_train)
        adj_val = utils.normalize_adj_tensor(adj_val)
        adj_test = utils.normalize_adj_tensor(adj_test)
    adj_train=SparseTensor(row=adj_train._indices()[0], col=adj_train._indices()[1],value=adj_train._values(), sparse_sizes=adj_train.size()).t()
    adj_val=SparseTensor(row=adj_val._indices()[0], col=adj_val._indices()[1],value=adj_val._values(), sparse_sizes=adj_val.size()).t()
    adj_test=SparseTensor(row=adj_test._indices()[0], col=adj_test._indices()[1],value=adj_test._values(), sparse_sizes=adj_test.size()).t()
    
    #teacher_model
    if args.teacher_model=='GCN':
        teacher_model = GCN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)
    elif args.teacher_model=='SGC':
        teacher_model = SGC_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm=None, sgc=True, act=args.activation).to(device)
    else:
        teacher_model = SAGE_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)   

    train_teacher()
    
  
    