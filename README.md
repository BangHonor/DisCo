# Disentangled Condensation for Graphs (DisCo)
Official codebase for paper Disentangled Condensation for Graphs. This codebase is based on the open-source Pytorch Geometric framework.

## Overview

**TLDR:** This paper introduces Disentangled Condensation (DisCo), a scalable graph condensation method for graphs of varying sizes. DisCo incorporates node and edge condensation modules to realize the condensation of nodes and edges in a disentangled manner. The node condensation module synthesizes condensed nodes that maintain a similar feature distribution to the original nodes, while the edge condensation module preserves the graph's topology structure. DisCo successfully scales up to the ogbn-papers100M graph with over 100 million nodes and 1 billion edges. Experimental results on various datasets demonstrate that DisCo outperforms state-of-the-art methods by a significant margin.

**Abstart:** Graph condensation has emerged as an intriguing technique to provide Graph Neural Networks (GNNs) for large-scale graphs with a more compact yet informative small graph to save the expensive costs of large-scale graph learning. Despite the promising results achieved, previous graph condensation methods often employ an entangled condensation strategy that involves condensing nodes and edges simultaneously, leading to substantial GPU memory demands. This entangled strategy has considerably impeded the scalability of graph condensation, impairing its capability to condense extremely large-scale graphs and produce condensed graphs with high fidelity. Therefore, this paper presents Disentangled Condensation for large-scale graphs, abbreviated as DisCo, to provide scalable graph condensation for graphs of varying sizes. At the heart of DisCo are two complementary components, namely node and edge condensation modules, that realize the condensation of nodes and edges in a disentangled manner. In the node condensation module, we focus on synthesizing condensed nodes that exhibit a similar node feature distribution to original nodes using a pre-trained node classification model while incorporating class centroid alignment and anchor attachment regularizers. After node condensation, in the edge condensation module, we preserve the topology structure by transferring the link prediction model of the original graph to the condensed nodes, generating the corresponding condensed edges. Based on the disentangled strategy, the proposed DisCo can successfully scale up to the ogbn-papers100M graph with over 100 million nodes and 1 billion edges with flexible reduction rates. Extensive experiments on five common datasets further demonstrate that the proposed DisCo yields results superior to state-of-the-art (SOTA) counterparts by a significant margin.

![Disco_framework 图标](https://github.com/BangHonor/DisCo/blob/main/Disco_framework.png)

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

## Reproduce
Please follow the instructions below to replicate the results in the paper.

Run to reproduce the results of Table 3 in "scripts/baseline_comparison.sh" . 

Run to reproduce the results of Table 5 in "scripts/papers100M.sh" . 

Run to reproduce the results of Table 6 in "scripts/generalizability.sh" . 

Run to reproduce the results of Table 7 in "scripts/nas.sh" . 

## Contact
Please feel free to contact me via email (xiaozhb@zju.edu.cn) if you are interested in my research :)
