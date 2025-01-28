# Disentangled Condensation for Large-scale Graphs (DisCo)
Official codebase for paper Disentangled Condensation for Large-scale Graphs. The paper can be found here: https://arxiv.org/abs/2401.12231. This codebase is based on the open-source Pytorch Geometric framework.

## Overview

**Abstart:** Graph condensation has emerged as an intriguing technique to save the expensive training costs of Graph Neural Networks~(GNNs) by substituting a condensed small graph with the original graph. Despite the promising results achieved, previous methods usually employ an entangled paradigm of redundant parameters (nodes, edges, GNNs), which incurs complex joint optimization during condensation. This paradigm has considerably impeded the scalability of graph condensation, making it challenging to condense extremely large-scale graphs and generate high-fidelity condensed graphs. Therefore, we propose to disentangle the condensation process into a two-stage GNN-free paradigm, independently condensing nodes and generating edges while eliminating the need to optimize GNNs at the same time. The node condensation module avoids the complexity of GNNs by focusing on node feature alignment with anchors of the original graph, while the edge translation module constructs the edges of the condensed nodes by transferring the original structure knowledge with neighborhood anchors. This simple yet effective approach achieves at least 10 times faster than state-of-the-art methods with comparable accuracy on medium-scale graphs. Moreover, the proposed DisCo can successfully scale up to the Ogbn-papers100M graph containing over 100 million nodes with flexible reduction rates and improves performance on the second-largest Ogbn-products dataset by over 5%. Extensive downstream tasks and ablation studies on five common datasets further demonstrate the effectiveness of the proposed DisCo framework.

![Disco_framework](https://github.com/BangHonor/DisCo/blob/main/DisCo_frameworkv3.png)

## Requirements
See requirments.txt file for more information about how to install the dependencies.

## Run the Code
For transductive setting, please run the following command:
```
python -u LargeScaleCondensing.py --dataset ogbn-arxiv --edge_pred aggr --condensing_loop 1500 --reduction_rate=0.01 --gpu_id=0 --model=GCN --seed=1
```
where the parameter ```r``` represents the ratio of condensed nodes to the labeled nodes of each class. For example, in the cora dataset, there are 140 labeled nodes, which accounts for 5.2% of the entire dataset. If we set r=0.5, it means that the condensed node number for each class will be 50% of the labeled nodes for that class. Consequently, the final reduction rate can be calculated as 5.2% * 0.5 = 2.6%. It is important to note that the parameter ```r``` differs from the actual reduction rate stated in the paper for the transductive setting. In our context, ```edge_pred``` refers to the link prediction method used, ```aggr``` represents our specific link prediction approach, and ```none```signifies the simple link prediction model as mentioned in the paper. The term `condensing_loop` denotes the number of epochs dedicated to node condensation. Lastly, `model` refers to the trained model employed on the condensed graph.

For inductive setting, please run the following command:
```
python -u LargeScaleCondensing_induct.py --dataset reddit --edge_pred aggr --condensing_loop 2500 --reduction_rate=0.002 --gpu_id=0 --model=GCN --seed=1
```

For the ogbn-papers100M dataset, please run the following command to make use of the super large-scale condensation:
```
 python -u LargeScaleCondensing_Sampled.py --dataset ogbn-papers100M --edge_pred aggr --condensing_loop 2500 --inference True  --reduction_rate=$0.01 --gpu_id=0 --model=SGC --seed=1
```

## Reproduce
Please follow the instructions below to replicate the results in the paper.

Run to reproduce the results of Table 3 in "scripts/baseline_comparison.sh" . The train_original.py and train_original_induct.py files aim to compute the whole dataset performance without graph condensation.

Run to reproduce the results of Table 5 in "scripts/papers100M.sh" . The train_coreset_papers100M.py file aims to test the coreset method on the ogbn-papers100M dataset.

Run to reproduce the results of Table 6 in "scripts/generalizability.sh" . 

Run to reproduce the results of Table 7 in "scripts/nas.sh" . The nas_transductive.py and nas_inductive.py files aim to test the coreset method on the ogbn-papers100M dataset.


## Contact
Please feel free to contact me via email (xiaozhb@zju.edu.cn) if you are interested in my research :)
