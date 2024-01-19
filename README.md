This repository is the implementation of [MHGAN]

## Installation

Install [pytorch](https://pytorch.org/get-started/locally/)

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

# Data Preprocessing
We used datasets from [Heterogeneous Graph Attention Networks](https://github.com/Jhy1993/HAN) (Xiao Wang et al.) and uploaded the preprocessing code of acm data as an example.
Preprocessed by GTN.
We use the version provided by GTN(https://github.com/seongjunyun/Graph_Transformer_Networks).

Take DBLP as an example to show the formats of input data:

`node_features.pkl` is a numpy array whose shape is (num_of_nodes, num_of_features). It contains input node features.

`edges.pkl` is a list of scipy sparse matrices. Each matrix has a shape of (num_of_nodes, num_of_nodes) and is formed by edges of a certain edge type.

`labels.pkl` is a list of lists. labels[0] is a list containing training labels and each item in it has the form [node_id, target]. labels[1] and labels[2] are validation labels and test labels respectively with the same format.

Note that the inputs of our method are only raw information of a heterogeneous network (network topology, node types, edge types, and node attributes if applicable). We do not need to manually design any meta path or meta graph.

## Running the code
``` 
$ mkdir data
$ cd data
```
Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view?usp=sharing) and extract data.zip into data folder.
```
$ cd ..
```
- ACM
- NodeAggregationConv Dropout=0.5  lr=0.002 
- SimilarAttentionConv Dropout=0.5  lr=0.01
- class=3
- L2—norm= 'False'
```
python ACM_run.py --adaptive_lr True --norm False
```
- IMDB
- NodeAggregationConv Dropout=0.5  lr=0.002 
- NSimilarAttentionConv Dropout=0.5  lr=0.01
- class=3
- L2—norm= 'True'
```
python IMDB_run.py --adaptive_lr True --norm True
```
- DBLP    
- NodeAggregationConv Dropout=0.5  lr=0.002  
- SimilarAttentionConv Dropout=0.2  lr=0.01
- class=4 
- L2—norm='True'
```
python DBLP_run.py --adaptive_lr True --norm True
```

