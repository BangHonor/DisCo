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
parser.add_argument('--dataset', type=str, default='amazon-products')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
#edge
parser.add_argument('--edge_pred', type=str, default='aggr')
parser.add_argument('--inference', type=bool, default=False)
#gnn
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--validation_model', type=str, default='MLP')
parser.add_argument('--model', type=str, default='GCN')
#ratio
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.01)
#condensation
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_teacher_model', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.001)
parser.add_argument('--feat_alpha', type=float, default=100)
parser.add_argument('--dis_alpha', type=float, default=2)
parser.add_argument('--anchor', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.99, help='adj threshold.')
parser.add_argument('--sample_num', type=int, default=5)
parser.add_argument('--save', type=int, default=1)
#loop and validation
parser.add_argument('--teacher_model_loop', type=int, default=600)
parser.add_argument('--condensing_loop', type=int, default=2500)#arxiv:1500 reddit/reddit2/products/amazon:2500 papers:5000
parser.add_argument('--student_model_loop', type=int, default=3000)
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


def generate_labels_syn(labels_train):
    from collections import Counter
    counter = Counter(labels_train.cpu().numpy())
    num_class_dict = {}

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    labels_syn = []
    syn_class_indices = {}

    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * args.reduction_rate)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

    return labels_syn, num_class_dict


def get_ini_feat(feat_train_batch):
    idx_selected = []

    from collections import Counter;
    counter = Counter(labels_syn.cpu().numpy())
    labels_train_batch_np = labels_train_batch.cpu().numpy()
    class_dict={}

    for i in range(nclass):
        class_dict['class_%s'%i] = (labels_train_batch_np == i)

    for c in range(nclass):
        tmp = retrieve_class(c, class_dict, num=counter[c])
        tmp = list(tmp)
        idx_selected = idx_selected + tmp
    idx_selected = np.array(idx_selected).reshape(-1)

    return feat_train_batch[idx_selected]
    

def retrieve_class(c, class_dict, num=256):
    idx = np.arange(len(labels_train_batch))
    idx = idx[class_dict['class_%s'%c]]
    return np.random.permutation(idx)[:num]
    

def link_prediction(pge_edge, nedges):
    start = time.perf_counter()
    optimizer_pge = optim.Adam(pge_edge.parameters(), lr=args.lr_adj)

    feat_transform=feat_train
    if args.edge_pred=='aggr':
        aggr=MessagePassing(aggr="max")
        aggr_adj=SparseTensor(row=adj_train._indices()[0], col=adj_train._indices()[1], value=adj_train._values(), sparse_sizes=adj_train.size()).t().cuda()
        if args.inference:
            loader=NeighborSampler(aggr_adj,
                node_idx=torch.arange(len(labels_train)),
                sizes=[-1], 
                batch_size=args.batch_size,
                num_workers=12, 
                return_e_id=False,
                num_nodes=len(labels_train),
                shuffle=False)
            xs: List[Tensor] = []
            for batch_size, n_id, batch_adj in loader:
                x = feat_transform[n_id].to(device)
                edge_index = batch_adj.adj_t.to(device)
                x = aggr.propagate(edge_index, x=x)[:batch_size]
                xs.append(x.cpu())
            feat_transform = torch.cat(xs, dim=0).cuda()
        else:    
            feat_transform=aggr.propagate(aggr_adj, x=feat_transform)
        feat_transform=torch.concat((feat_train, feat_transform), dim=1)
        torch.save(feat_transform, f'{root}/temp/feat_transform_aggr_max_{args.dataset}_{args.seed}.pt')

    edge_index = adj_train._indices()
    best_acc=0
    neg_edge_index = negative_sampling(edge_index.cpu(), feat_train.shape[0], len(edge_index[0])).cpu()

    for i in range(30000):
        index = torch.randint(0, len(edge_index[0]), (nedges,))
        neg_index = torch.randint(0, len(neg_edge_index[0]), (3 * nedges,))
        pos_edge_embed = torch.concat((feat_transform[edge_index[0][index]], feat_transform[edge_index[1][index]]), dim=1).cpu()
        neg_edge_embed = torch.concat((feat_transform[neg_edge_index[0][neg_index]], feat_transform[neg_edge_index[1][neg_index]]), dim=1).cpu()
        edge_embed_sample = torch.concat((pos_edge_embed, neg_edge_embed), dim = 0).cuda()
        y = torch.concat((torch.ones(nedges), torch.zeros(3 * nedges))).cuda()

        optimizer_pge.zero_grad()
        pred = pge_edge.forward(edge_embed_sample)
        criterion = nn.BCELoss()
        loss = criterion(pred, y.float())
        loss.backward()
        optimizer_pge.step()

        if i%500==0:
            y_pred = torch.round(pred)
            accuracy = torch.eq(y_pred, y).sum().item()/len(y)
            recall = recall_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            precision = precision_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()) 

            if accuracy>best_acc:
                best_acc=accuracy
                if args.edge_pred=='aggr':
                    torch.save(pge_edge.state_dict(), f'{root}/saved_ours_large/pge_aggr_max_{args.dataset}_{args.seed}.pt')
                else:
                    torch.save(pge_edge.state_dict(), f'{root}/saved_ours_large/pge_{args.dataset}_{args.seed}.pt')
            print("Acc:", accuracy, "Recall:", recall, "Precision:", precision)

            if i%10000==0:
                neg_edge_index = negative_sampling(edge_index.cpu(), feat_train.shape[0], len(edge_index[0])).cpu()

    end = time.perf_counter()
    print('Link Prediction Model Pre-training Duration:', round(end-start), 's')
    return 


def node_condensation():
    start = time.perf_counter()
    validation_model = MLP_PYG(channel_list=[d, args.hidden, args.hidden, args.hidden, nclass], dropout=[args.dropout, args.dropout, args.dropout, args.dropout], num_layers=4, norm='BatchNorm', act='relu').to(device)
    validation_model.initialize()
    validation_model.train()
    optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
    optimizer = optim.Adam(validation_model.parameters(), lr=args.lr_teacher_model, weight_decay=1e-5)

    # Inversion
    if not os.path.exists(root+'/saved_model_large/teacher/MLP_4_'+args.dataset+'_sampled_'+str(component)+'_'+str(args.sample_num)+'_'+str(args.seed)+'.pt'):
        for i in range(args.teacher_model_loop):
            optimizer.zero_grad()
            output = validation_model.forward(feat_train_batch)
            loss = F.nll_loss(output, labels_train_batch)
            loss.backward()
            optimizer.step()
        torch.save(validation_model.state_dict(), f'{root}/saved_model_large/teacher/MLP_4_{args.dataset}_sampled_{component}_{args.sample_num}_{args.seed}.pt')

    validation_model.load_state_dict(torch.load(f'{root}/saved_model_large/teacher/MLP_4_{args.dataset}_sampled_{component}_{args.sample_num}_{args.seed}.pt'))
    output = validation_model.predict(feat_test)
    acc_test = utils.accuracy(output, labels_test)
    print("MLP Test Acc:", acc_test)

    for i in range(args.condensing_loop+1):
        validation_model.train()
        optimizer_feat.zero_grad()
        output_syn_batch = validation_model.forward(feat_syn)
        loss = F.nll_loss(output_syn_batch, labels_syn)

        #alignment loss
        feat_loss=torch.tensor(0.0).to(device)
        dis_loss=torch.tensor(0.0).to(device)
        loss_fn=nn.MSELoss()
        for c in range(nclass):
            if coeff[c]>0:
                feat_loss += (coeff[c] * loss_fn(feat_train_batch[index[c]].mean(dim=0), feat_syn[index_syn[c]].mean(dim=0)))
                _, I = knn_class[c].search(feat_syn[index_syn[c]].detach().cpu().numpy(), args.anchor)
                dis_loss += (coeff[c] * loss_fn(feat_syn[index_syn[c]], feat_train_batch[index[c]][I.ravel()]))

        feat_loss = feat_loss / coeff_sum
        dis_loss = dis_loss / coeff_sum
        loss += (args.feat_alpha * feat_loss + args.dis_alpha * dis_loss)
        loss.backward()
        optimizer_feat.step()

        if i%100 == 0:
            output_syn = validation_model.predict(feat_syn)
            acc_test = utils.accuracy(output_syn, labels_syn)
            print("Epoch", i, ", Syn Test Acc:", acc_test)

    torch.save(feat_syn, f'{root}/saved_ours_large/feat_{args.dataset}_{args.anchor}_{args.reduction_rate}_sampled_{component}_{args.sample_num}_{args.seed}.pt')
    torch.save(labels_syn, f'{root}/saved_ours_large/labels_{args.dataset}_{args.anchor}_{args.reduction_rate}_sampled_{component}_{args.sample_num}_{args.seed}.pt')
    end = time.perf_counter()
    print('Node Condensation Duration:', round(end-start), 's')


def edge_construction():
    if args.edge_pred=='aggr':# find the anchors，directly use their feat_transform
        feat_syn_neighbor = torch.zeros_like(feat_syn)
        feat_transform = torch.load(f'{root}/temp/feat_transform_aggr_max_{args.dataset}_{args.seed}.pt').cuda()
        if args.model == 'GIN':
            neighbor = 100
        else:
            neighbor = 3
        for c in range(nclass):
            if c in num_class_dict:
                _, anchor = knn_class[c].search(feat_syn[index_syn[c]].cpu().numpy(), neighbor)
                feat_syn_neighbor[index_syn[c]] = feat_transform[index[c]][anchor,:d].max(dim=1).values
        feat_syn_transform = torch.concat((feat_syn, feat_syn_neighbor),dim=1)
    else:
        feat_syn_transform = feat_syn

    start = time.perf_counter()
    row = min(10000, n)
    adj_syn=torch.zeros(row, n).to(device)
    edge_index_syn=[]
    edge_weight_syn=[]
    for i in range(n-1):
        loop=math.floor(i/row)
        edge_row=pge_edge.inference(torch.cat([feat_syn_transform[[i]*(n-i-1)], feat_syn_transform[np.arange(i+1,n)]], axis=1))
        edge_row_reverse=pge_edge.inference(torch.cat([feat_syn_transform[np.arange(i+1,n)], feat_syn_transform[[i]*(n-i-1)]], axis=1))
        adj_syn[i%row][i+1:]=(edge_row+edge_row_reverse)/2
        if (i%row==row-1) or i==n-2:
            adj_syn[adj_syn<args.threshold]=0
            adj_index=torch.nonzero(adj_syn).T
            edge_index_syn.append(torch.stack([adj_index[0]+torch.full((adj_index.shape[1],), loop*row).to(device),adj_index[1]]))
            edge_weight_syn.append(adj_syn[adj_index[0], adj_index[1]])
            adj_syn=torch.zeros(row, n).to(device)
    edge_index_syn=torch.cat(edge_index_syn, dim=1)
    edge_weight_syn=torch.cat(edge_weight_syn)
    edge_index_syn, edge_weight_syn = to_undirected(edge_index_syn, edge_weight_syn)
    edge_index_syn, edge_weight_syn = add_self_loops(edge_index_syn, edge_weight_syn, num_nodes=n)
    end = time.perf_counter()
    print("Edge Construction Time:", end-start)

    torch.save(edge_index_syn, f'{root}/saved_ours_large/edge_index_syn_{args.dataset}_{args.reduction_rate}_sampled_{args.sample_num}_{args.seed}.pt')
    torch.save(edge_weight_syn, f'{root}/saved_ours_large/edge_weight_syn_{args.dataset}_{args.reduction_rate}_sampled_{args.sample_num}_{args.seed}.pt')

    return edge_index_syn, edge_weight_syn


def train_on_syn_graph():
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=1e-5)

    best_val=0
    best_test=0
    print("Traingin Model on the Condensed Graph!")
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

        if j % args.student_val_stage == 0:
            if args.inference == False:
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
                    if args.edge_pred=='aggr':
                        torch.save(model.state_dict(), f'{root}/saved_model_large/student/{args.dataset}_aggr_{args.model}_{args.reduction_rate}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt')
                    else:
                        torch.save(model.state_dict(), f'{root}/saved_model_large/student/{args.dataset}_{args.model}_{args.reduction_rate}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt')
    end = time.perf_counter()
    print('Model Training Duration:', round(end-start), 's')
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
    if args.edge_pred=='aggr':
        pge_edge = PGE_Edge(nfeat=2*d, device=device, args=args).cuda()
    else:
        pge_edge = PGE_Edge(nfeat=d, device=device, args=args).cuda()

    if args.edge_pred=='aggr':
        if not os.path.exists(root+'/saved_ours_large/pge_aggr_max_'+args.dataset+'_'+str(args.seed)+'.pt'):
            print("Pretraining Link Prediction Model!")
            link_prediction(pge_edge, 10000)
        pge_edge.load_state_dict(torch.load(f'{root}/saved_ours_large/pge_aggr_max_{args.dataset}_{args.seed}.pt'))
    else:
        if not os.path.exists(root+'/saved_ours_large/pge_'+args.dataset+'_'+str(args.seed)+'.pt'):
            print("Pretraining Link Prediction Model!")
            link_prediction(pge_edge, 10000)
        pge_edge.load_state_dict(torch.load(f'{root}/saved_ours_large/pge_{args.dataset}_{args.seed}.pt'))

    node_per_component = math.ceil(feat_train.shape[0] / args.sample_num)
    for component in range(args.sample_num):
        if os.path.exists(root+'/saved_ours_large/feat_'+args.dataset+'_'+str(args.anchor)+'_'+str(args.reduction_rate)+'_sampled_'+str(component)+'_'+str(args.sample_num)+'_'+str(args.seed)+'.pt'):
            continue

        batch = torch.arange(node_per_component * component, min(node_per_component * (component + 1), feat_train.shape[0]))
        feat_train_batch = feat_train[batch]
        labels_train_batch = labels_train[batch]

        labels_syn, num_class_dict = generate_labels_syn(labels_train_batch)
        labels_syn = torch.LongTensor(labels_syn).cuda()
        n = len(labels_syn)
        feat_syn = nn.Parameter(torch.FloatTensor(n, d).cuda())
        feat_syn.data.copy_(get_ini_feat(feat_train_batch))

        index=[]
        index_syn=[]
        coeff=[]
        coeff_sum=0
        for c in range(nclass):
            index.append(torch.where(labels_train_batch==c))
            index_syn.append(torch.where(labels_syn==c))
            if c in num_class_dict:
                coe = num_class_dict[c] / max(num_class_dict.values())
                coeff_sum += coe
                coeff.append(coe)
            else:
                coeff.append(0)
        coeff_sum=torch.tensor(coeff_sum).to(device)

        knn_class=[]
        for c in range(nclass):
            if c in num_class_dict:
                knn = faiss.IndexFlatL2(d)
                knn.add(feat_train_batch[index[c]].cpu().numpy())
                knn_class.append(knn)
            else:
                knn_class.append(0)

        print("Component:", component)
        node_condensation()

    feat_syn=torch.load(f'{root}/saved_ours_large/feat_{args.dataset}_{args.anchor}_{args.reduction_rate}_sampled_0_{args.sample_num}_{args.seed}.pt').detach().cuda()
    labels_syn=torch.load(f'{root}/saved_ours_large/labels_{args.dataset}_{args.anchor}_{args.reduction_rate}_sampled_0_{args.sample_num}_{args.seed}.pt').detach().cuda()
    for i in range(1, args.sample_num):
        feat_syn_temp=torch.load(f'{root}/saved_ours_large/feat_{args.dataset}_{args.anchor}_{args.reduction_rate}_sampled_{i}_{args.sample_num}_{args.seed}.pt').detach().cuda()
        feat_syn=torch.concat((feat_syn, feat_syn_temp), dim=0)
        labels_syn_temp=torch.load(f'{root}/saved_ours_large/labels_{args.dataset}_{args.anchor}_{args.reduction_rate}_sampled_{i}_{args.sample_num}_{args.seed}.pt').detach().cuda()
        labels_syn=torch.concat((labels_syn, labels_syn_temp))
    n = len(labels_syn)    

    index=[]
    index_syn=[]
    _, num_class_dict = generate_labels_syn(labels_train)
    for c in range(nclass):
        index.append(torch.where(labels_train==c))
        index_syn.append(torch.where(labels_syn==c))
    
    knn_class=[]
    for c in range(nclass):
        if c in num_class_dict:
            knn = faiss.IndexFlatL2(d)
            knn.add(feat_train[index[c]].cpu().numpy())
            knn_class.append(knn)
        else:
            knn_class.append(0)

    #edge construction
    edge_index_syn, edge_weight_syn = edge_construction()
    if args.model in ['GCN', 'SGC', 'JKNet']:
        edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, n, add_self_loops=False)
        
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

    #train on the synthetic graph
    train_on_syn_graph()