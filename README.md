# Disentangled Condensation for Graphs (DisCo)
Official codebase for paper Disentangled Condensation for Graphs. This codebase is based on the open-source Pytorch Geometric framework.

## Overall Framework
**Abstart:** Graph condensation has emerged as an intriguing technique to provide Graph Neural Networks (GNNs) for large-scale graphs with a more compact yet informative small graph to save the expensive costs of large-scale graph learning. Despite the promising results achieved, previous graph condensation methods often employ an entangled condensation strategy that involves condensing nodes and edges simultaneously, leading to substantial GPU memory demands. This entangled strategy has considerably impeded the scalability of graph condensation, impairing its capability to condense extremely large-scale graphs and produce condensed graphs with high fidelity. Therefore, this paper presents Disentangled Condensation for large-scale graphs, abbreviated as DisCo, to provide scalable graph condensation for graphs of varying sizes. At the heart of DisCo are two complementary components, namely node and edge condensation modules, that realize the condensation of nodes and edges in a disentangled manner. In the node condensation module, we focus on synthesizing condensed nodes that exhibit a similar node feature distribution to original nodes using a pre-trained node classification model while incorporating class centroid alignment and anchor attachment regularizers. After node condensation, in the edge condensation module, we preserve the topology structure by transferring the link prediction model of the original graph to the condensed nodes, generating the corresponding condensed edges. Based on the disentangled strategy, the proposed DisCo can successfully scale up to the ogbn-papers100M graph with over 100 million nodes and 1 billion edges with flexible reduction rates. Extensive experiments on five common datasets further demonstrate that the proposed DisCo yields results superior to state-of-the-art (SOTA) counterparts by a significant margin.

![Disco_framework 图标](https://github.com/BangHonor/DisCo/blob/main/Disco_framework.png)

## Prerequisites
**Install dependencies**

See requirments.txt file for more information about how to install the dependencies.
