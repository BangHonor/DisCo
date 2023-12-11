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
parser.add_argument('--dataset', type=str, default='cora')
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
parser.add_argument('--reduction_rate', type=float, default=0.25)
#condensation
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_teacher_model', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--feat_alpha', type=float, default=100)
parser.add_argument('--dis_alpha', type=float, default=2)
parser.add_argument('--anchor', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.99, help='adj threshold.')
parser.add_argument('--save', type=int, default=1)
#loop and validation
parser.add_argument('--teacher_model_loop', type=int, default=600)
parser.add_argument('--condensing_loop', type=int, default=1500)#arxiv:1500 reddit/reddit2/products/amazon:2500 papers:5000
parser.add_argument('--student_model_loop', type=int, default=5000)
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


def generate_labels_syn():
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


def get_ini_feat(feat_train):
    idx_selected = []

    from collections import Counter;
    counter = Counter(labels_syn.cpu().numpy())
    labels_train_np = labels_train.cpu().numpy()
    class_dict={}

    for i in range(nclass):
        class_dict['class_%s'%i] = (labels_train_np == i)

    for c in range(nclass):
        tmp = retrieve_class(c, class_dict, num=counter[c])
        tmp = list(tmp)
        idx_selected = idx_selected + tmp
    idx_selected = np.array(idx_selected).reshape(-1)

    return feat_train[idx_selected]
    

def retrieve_class(c, class_dict, num=256):
    idx = np.arange(len(labels_train))
    idx = idx[class_dict['class_%s'%c]]
    return np.random.permutation(idx)[:num]
    

def link_prediction(pge_edge, nedges):
    start = time.perf_counter()
    optimizer_pge = optim.Adam(pge_edge.parameters(), lr=args.lr_adj)

    feat_transform=feat
    if args.edge_pred=='aggr':
        aggr=MessagePassing(aggr="max")
        aggr_adj=SparseTensor(row=adj._indices()[0], col=adj._indices()[1], value=adj._values(), sparse_sizes=adj.size()).t().cuda()
        if args.inference:
            loader=NeighborSampler(aggr_adj,
                node_idx=torch.arange(len(labels)),
                sizes=[-1], 
                batch_size=args.batch_size,
                num_workers=12, 
                return_e_id=False,
                num_nodes=len(labels),
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
        feat_transform=torch.concat((feat, feat_transform), dim=1)
        torch.save(feat_transform, f'{root}/temp/feat_transform_aggr_max_{args.dataset}_{args.seed}.pt')

    edge_index = adj._indices()
    best_acc = 0
    neg_edge_index = negative_sampling(edge_index, feat.shape[0], 3 * len(edge_index[0])).cpu()

    if args.dataset == 'ogbn-arxiv':
        epoch = 10000
    elif args.dataset == 'ogbn-products':
        epoch = 20000
    else:
        epoch = 500

    for i in range(epoch):#arxiv:10000 products:20000 amazon:30000 
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

        if i%100==0:
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

    end = time.perf_counter()
    print('Link Prediction Model Pre-training Duration:', round(end-start), 's')
    return 


def node_condensation():
    start = time.perf_counter()
    if args.dataset == 'cora':
        validation_model = MLP_PYG(channel_list=[d, args.hidden, args.hidden, nclass], num_layers=3, act='relu').to(device)
    else:
        validation_model = MLP_PYG(channel_list=[d, args.hidden, args.hidden, args.hidden, nclass], dropout=[args.dropout, args.dropout, args.dropout, args.dropout], num_layers=4, norm='BatchNorm', act='relu').to(device)
    validation_model.initialize()
    validation_model.train()
    optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
    optimizer = optim.Adam(validation_model.parameters(), lr=args.lr_teacher_model, weight_decay=1e-5)

    # Inversion
    if not os.path.exists(root+'/saved_model_large/teacher/MLP_4_'+args.dataset+'_'+str(args.seed)+'.pt'):
        for i in range(args.teacher_model_loop):
            optimizer.zero_grad()
            output = validation_model.forward(feat_train)
            loss = F.nll_loss(output, labels_train)
            loss.backward()
            optimizer.step()
    torch.save(validation_model.state_dict(), f'{root}/saved_model_large/teacher/MLP_4_{args.dataset}_{args.seed}.pt')
    validation_model.load_state_dict(torch.load(f'{root}/saved_model_large/teacher/MLP_4_{args.dataset}_{args.seed}.pt'))
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
                feat_train_c=feat_train[index[c]]
                feat_syn_c=feat_syn[index_syn[c]]
                feat_loss += (coeff[c] * loss_fn(feat_train_c.mean(dim=0), feat_syn_c.mean(dim=0)))
                # if feat_syn_c.shape[0] > 1:
                #     feat_loss += (coeff[c] * loss_fn(feat_train_c.std(dim=0), feat_syn_c.std(dim=0)))
                _, I = knn_class[c].search(feat_syn_c.detach().cpu().numpy(), args.anchor)
                I = I.ravel()
                # dis_loss += (coeff[c] * loss_fn(feat_syn_c, feat_train_c[I]))
                dis_loss += (coeff[c] * loss_fn(feat_syn_c, feat_train_c[I].view(feat_syn_c.shape[0], args.anchor, d).mean(dim=1)))
        feat_loss = feat_loss / coeff_sum
        dis_loss = dis_loss / coeff_sum
        loss += (args.feat_alpha * feat_loss + args.dis_alpha * dis_loss)
        loss.backward()
        optimizer_feat.step()

        if i%100 == 0:
            output_syn = validation_model.predict(feat_syn)
            acc_test = utils.accuracy(output_syn, labels_syn)
            print("Epoch", i, ", Syn Test Acc:", acc_test)

    torch.save(feat_syn, f'{root}/saved_ours_large/feat_{args.dataset}_{args.anchor}_{args.reduction_rate}_{args.seed}.pt')
    end = time.perf_counter()
    print('Node Condensation Duration:', round(end-start), 's')


def edge_construction():
    adj_syn=torch.zeros(n, n)
    if args.edge_pred=='aggr':#find the anchors，directly use their feat_transform
        feat_syn_neighbor = torch.zeros_like(feat_syn)
        feat_transform = torch.load(f'{root}/temp/feat_transform_aggr_max_{args.dataset}_{args.seed}.pt').cuda()[idx_train]
        neighbor = 3
        for c in range(nclass):
            if c in num_class_dict:
                _, anchor = knn_class[c].search(feat_syn[index_syn[c]].cpu().numpy(), neighbor)
                feat_syn_neighbor[index_syn[c]] = feat_transform[index[c]][anchor,:d].max(dim=1).values
        feat_syn_transform = torch.concat((feat_syn, feat_syn_neighbor),dim=1)
    else:
        feat_syn_transform = feat_syn

    for i in range(n):
        adj_syn[i]=pge_edge.inference(torch.cat([feat_syn_transform[[i]*n], feat_syn_transform[np.arange(n)]], axis=1)).detach()
    adj_syn=(adj_syn+adj_syn.T)/2
    adj_syn[adj_syn<args.threshold]=0
    adj_syn.diagonal().fill_(1)
    edge_index_syn=torch.nonzero(adj_syn).T.detach().cuda()
    edge_weight_syn=adj_syn[edge_index_syn[0], edge_index_syn[1]].detach().cuda()

    return edge_index_syn, edge_weight_syn

def train_on_syn_graph():
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=1e-5)
    if args.inference:
        inference_loader=NeighborSampler(adj,
            sizes=[-1], 
            batch_size=args.batch_size,
            num_workers=12, 
            return_e_id=False,
            num_nodes=len(labels),
            shuffle=False
        )

    best_val = 0
    best_test = 0
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
                    output = model.predict(feat.to(device), adj.to(device))
                else:
                    output = model.predict(feat.to(device))
            else:
                if args.model!='MLP':
                    output = model.inference(feat, inference_loader, device)
                else:
                    output = model.inference(feat, batch_size = 500000)

            acc_train = utils.accuracy(output[idx_train], labels_train)
            acc_val = utils.accuracy(output[idx_val], labels_val)
            acc_test = utils.accuracy(output[idx_test], labels_test)
            
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
    feat = torch.FloatTensor(data.features).detach().cuda()
    adj = utils.to_tensor(data.adj, device='cpu').detach()
    labels = torch.LongTensor(data.labels).cuda()
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    feat_train, feat_val, feat_test = feat[idx_train], feat[idx_val], feat[idx_test]
    labels_train, labels_val, labels_test = labels[idx_train], labels[idx_val], labels[idx_test]
    d = feat.shape[1]
    nclass= int(labels.max()+1)
    if args.edge_pred in ['aggr']:
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
    
    labels_syn, num_class_dict = generate_labels_syn()
    labels_syn = torch.LongTensor(labels_syn).cuda()
    n = len(labels_syn)
    feat_syn = nn.Parameter(torch.FloatTensor(n, d).cuda())
    feat_syn.data.copy_(get_ini_feat(feat_train))

    index=[]
    index_syn=[]
    coeff=[]
    coeff_sum=0
    for c in range(nclass):
        index.append(torch.where(labels_train==c))
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
            knn.add(feat_train[index[c]].cpu().numpy())
            knn_class.append(knn)
        else:
            knn_class.append(0)

    if not os.path.exists(root+'/saved_ours_large/feat_'+args.dataset+'_'+str(args.anchor)+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'.pt'):
        print("Node Condensation!")
        node_condensation()
    feat_syn=torch.load(f'{root}/saved_ours_large/feat_{args.dataset}_{args.anchor}_{args.reduction_rate}_{args.seed}.pt', map_location="cpu").detach().cuda() 

    #edge construction
    edge_index_syn, edge_weight_syn = edge_construction()
    if args.model in ['GCN', 'SGC', 'JKNet']:
        edge_index_syn, edge_weight_syn=utils.gcn_norm(edge_index_syn, edge_weight_syn, n, add_self_loops=False)
    
    if args.model=='GCN':
        if args.dataset=='cora':
            model = GCN_PYG(nfeat=d, nhid=128, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, act=args.activation).cuda()
        else:
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
        if not os.path.exists(root+'/temp/adj_norm_'+args.dataset+'_'+str(args.seed)+'.pt'):
            if utils.is_sparse_tensor(adj):
                adj = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj = utils.normalize_adj_tensor(adj)
            torch.save(adj, f'{root}/temp/adj_norm_{args.dataset}_{args.seed}.pt')
        else:
            adj = torch.load(f'{root}/temp/adj_norm_{args.dataset}_{args.seed}.pt').detach()
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],value=adj._values(), sparse_sizes=adj.size()).t()
    else:#add self loops
        adj = SparseTensor(row=torch.concat((adj._indices()[0], torch.arange(len(labels)))), col=torch.concat((adj._indices()[1],torch.arange(len(labels)))), value=torch.concat((adj._values(),torch.ones((len(labels),)))), sparse_sizes=adj.size()).t()

    #train on the synthetic graph
    train_on_syn_graph()
