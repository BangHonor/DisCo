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
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.conv import MessagePassing
from sklearn.metrics import recall_score, precision_score
from models.basicgnn_large import GCN as GCN_PYG, GIN as GIN_PYG, SGC as SGC_PYG, GraphSAGE as SAGE_PYG, JKNet as JKNet_PYG
from models.mlp import MLP as MLP_PYG
from models.parametrized_adj_lp import PGE_Edge

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--parallel_gpu_ids', type=list, default=[0, 1, 2], help='gpu id')
parser.add_argument('--dataset', type=str, default='ogbn-papers100M')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--method', type=str, default='herding')
#edge
parser.add_argument('--edge_pred', type=str, default='aggr')
parser.add_argument('--inference', type=bool, default=True)
#gnn
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--validation_model', type=str, default='MLP')
parser.add_argument('--model', type=str, default='SGC')
#ratio
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.05)
#condensation
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_teacher_model', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--feat_alpha', type=float, default=100)
parser.add_argument('--dis_alpha', type=float, default=2)
parser.add_argument('--anchor', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.999, help='adj threshold.')
parser.add_argument('--sample_num', type=int, default=2)
parser.add_argument('--save', type=int, default=1)
#loop and validation
parser.add_argument('--teacher_model_loop', type=int, default=1000)
parser.add_argument('--condensing_loop', type=int, default=2500)#arxiv:1500 reddit/reddit2/products/amazon:2500 papers:5000
parser.add_argument('--student_model_loop', type=int, default=2000)
parser.add_argument('--student_val_stage', type=int, default=100)
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


def train_on_coreset():
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=1e-5)

    val_inference_loader=NeighborSampler(
        edge_index=adj,
        sizes=[-1, -1], 
        node_idx=idx_val,
        batch_size=args.batch_size,
        num_workers=12, 
        return_e_id=False,
        num_nodes=N,
        shuffle=False
    )
    test_inference_loader=NeighborSampler(
        edge_index=adj,
        sizes=[-1, -1], 
        node_idx=idx_test,
        batch_size=args.batch_size,
        num_workers=12, 
        return_e_id=False,
        num_nodes=N,
        shuffle=False
    )

    best_val=0
    best_test=0
    start = time.perf_counter()
    for j in range(args.student_model_loop+1):
        model.train()
        optimizer.zero_grad()
        if args.model!='MLP':
            output_syn = model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        else:
            output_syn = model.forward(feat_syn)
        loss = F.nll_loss(output_syn, labels_syn)
        loss.backward()
        optimizer.step()

        if j%args.student_val_stage==0:
            if args.model!='MLP':
                output_val = model.large_inference(torch.FloatTensor(feat).to('cpu'), val_inference_loader, device)
                output_test = model.large_inference(torch.FloatTensor(feat).to('cpu'), val_inference_loader, device)
            else:
                output_val = model.inference(torch.FloatTensor(feat[idx_val]).to(device), batch_size = 500000)
                output_test = model.inference(torch.FloatTensor(feat[idx_test]).to(device), batch_size = 500000)
            acc_val = utils.accuracy(output_val, labels_val)
            acc_test = utils.accuracy(output_test, labels_test) 
            
            print(f'Epoch: {j:02d}, '
                    f'Valid: {100 * acc_val.item():.2f}%,'
                    f'Test: {100 * acc_test.item():.2f}%')
            
            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                if args.save:
                    torch.save(model.state_dict(), f'{root}/saved_model_large/student/{args.dataset}_random_{args.model}_{args.reduction_rate}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt')
    end = time.perf_counter()
    print('Model Training Duration:', round(end-start), 's')
    print("Best Test Acc:", best_test)


if __name__ == '__main__':
    root=os.path.abspath(os.path.dirname(__file__))

    dataset = PygNodePropPredDataset(name=args.dataset, root=root+'/dataset')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    idx_train, idx_val, idx_test = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([idx_train, idx_val, idx_test])
    N = data.num_nodes

    data.edge_index, _ = dropout_adj(data.edge_index, p = 0.4)
    data.edge_index = to_undirected(data.edge_index, num_nodes = N)

    # feat = np.memmap(root+'/dataset/ogbn_papers100M/raw/node_feat.npy', mode='r', shape=(111059956,512))
    feat = data.x.numpy()
    feat_train, feat_val, feat_test = torch.FloatTensor(feat[idx_train]).cuda(), torch.FloatTensor(feat[idx_val]).cuda(), torch.FloatTensor(feat[idx_test]).cuda()
    labels_train, labels_val, labels_test=torch.LongTensor((data.y[idx_train].numpy())).ravel().cuda(), torch.LongTensor((data.y[idx_val].numpy())).ravel().cuda(), torch.LongTensor((data.y[idx_test].numpy())).ravel().cuda()
    d = feat_train.shape[1]
    nclass= 172
    n = int(args.reduction_rate*feat_train.shape[0])

    from collections import Counter
    counter = Counter(labels_train.cpu().numpy())
    num_class_dict = {}
    n = len(labels_train)
    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    labels_syn = []
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * args.reduction_rate) - sum_
            labels_syn += [c] * num_class_dict[c]
        else:
            num_class_dict[c] = max(int(num * args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            labels_syn += [c] * num_class_dict[c]

    if args.method=='random':
        index=torch.randint(0, feat_train.shape[0], (n,))
    elif args.method=='herding':
        idx_selected = []
        for class_id, cnt in num_class_dict.items():
            idx = torch.where(labels_train==class_id)[0]
            features = feat_train[idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()

            for i in range(cnt):
                det = mean*(i+1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                id_min = torch.argmin(dis)
                selected.append(idx_left[id_min])
                del idx_left[id_min]
            idx_selected.append(idx[selected])
        index = torch.hstack(idx_selected)
    else:
        idx_selected = []
        for class_id, cnt in num_class_dict.items():
            idx = torch.where(labels_train==class_id)[0]
            feature = feat_train[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            dis = torch.cdist(feature, mean)[:,0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            for i in range(cnt-1):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max = torch.argmax(dis_min).item()
                idx_centers.append(id_max)
            idx_centers = np.array(idx_centers)
            idx_selected.append(idx[idx_centers])
        index = torch.hstack(idx_selected)

    feat_syn=feat_train[index].cuda()
    labels_syn=labels_train[index].cuda()
    adj_syn = sp.csr_matrix((np.ones(data.edge_index.shape[1]),(data.edge_index[0], data.edge_index[1])), shape=(N, N))[np.ix_(idx_train[index], idx_train[index])] 
    adj_syn = utils.to_tensor(adj_syn)
    edge_index_syn = adj_syn._indices().cuda()

    if args.model in ['GCN', 'SGC', 'JKNet']:
        edge_index_syn, edge_weight_syn=utils.gcn_norm(edge_index=edge_index_syn, edge_weight=None, num_nodes=n, add_self_loops=False)
        
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
        if not os.path.exists(root+'/temp/edge_index_norm_'+args.dataset+'_'+str(args.seed)+'.pt'):
            data.edge_index, edge_weight = utils.gcn_norm(data.edge_index, edge_weight = None, num_nodes = N, add_self_loops=False)
            torch.save(data.edge_index, f'{root}/temp/edge_index_norm_{args.dataset}_{args.seed}.pt')
            torch.save(edge_weight, f'{root}/temp/edge_weight_norm_{args.dataset}_{args.seed}.pt')
        else:
            data.edge_index = torch.load(f'{root}/temp/edge_index_norm_{args.dataset}_{args.seed}.pt')
            edge_weight = torch.load(f'{root}/temp/edge_weight_norm_{args.dataset}_{args.seed}.pt')
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=edge_weight, sparse_sizes=(N,N)).t().cpu()
    else:#add self loops
        adj = SparseTensor(row=torch.concat((data.edge_index[0], torch.arange(N))), col=torch.concat((data.edge_index[1], torch.arange(N))), value=torch.concat((torch.ones((data.edge_index.shape[1],)),torch.ones((N,)))), sparse_sizes=(N,N)).t().cpu()
    del data

    #train on the synthetic graph
    train_on_coreset()
