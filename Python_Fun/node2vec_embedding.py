#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:04:34 2023

@author: isaac
"""

from torch_geometric.nn import Node2Vec
import pandas as pd
import os.path as osp
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_networkx, to_scipy_sparse_matrix
from torch_geometric.utils import to_dense_adj
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def n2v_embedding(dataset_all,device):
    #%% Use only one moleculas compound
    n2v_mat=[]
    for sub in range(len(dataset_all)):
        fig,ax=plt.subplots(1,2,figsize=(18,15))
        data=dataset_all[sub]
        # print(data)
        # extract edge list
        edge_list = data.edge_index.t().numpy()
        adj=to_dense_adj(data.edge_index, edge_attr=data.edge_attr)
        adj=np.squeeze(adj.numpy())
        sns.heatmap(adj,cmap='jet',ax=ax[0])
        # print(edge_list[0:10])
        
        #%%
        # extract edge attributes
        edge_attr = data.edge_attr.numpy()
        # print(edge_attr[0:10])
        
        
        
        #%%
        #build the graph with
        #train_mask, test_mask, val_mask
        
        # np.random.seed(10)
        # get the nodes
        nodes = data.edge_index.t().numpy()
        nodes = np.unique(list(nodes[:,0]) + list(nodes[:,1]))
        
        np.random.shuffle(nodes) # shuffle node order
        # print(len(nodes))
        
        #%%
        # get train test and val sizes: (70% - 15% - 15%)
        train_size = int(len(nodes)*0.9)
        test_size = int(len(nodes)*0.95) - train_size
        val_size = len(nodes) - train_size - test_size
        
        # get train test and validation set of nodes
        train_set = nodes[0:train_size]
        test_set = nodes[train_size:train_size+test_size]
        val_set = nodes[train_size+test_size:]
        
        
        # print(len(train_set),len(test_set),len(val_set))
        # print(len(train_set)+len(test_set)+len(val_set) == len(nodes))
        
        # print("train set\t",train_set[:10])
        # print("test set \t",test_set[:10])
        # print("val set  \t",val_set[:10])
        
        #%% build test train val masks
        
        train_mask = torch.zeros(len(nodes),dtype=torch.long, device=device)
        for i in train_set:
            train_mask[i] = 1.
        
        test_mask = torch.zeros(len(nodes),dtype=torch.long, device=device)
        for i in test_set:
            test_mask[i] = 1.
            
        val_mask = torch.zeros(len(nodes),dtype=torch.long, device=device)
        for i in val_set:
            val_mask[i] = 1.
            
        # print("train mask \t",train_mask[0:15])
        # print("test mask  \t",test_mask[0:15])
        # print("val mask   \t",val_mask[0:15]) 
        
        #%% remove from the data what do we not use.
        
        # print("befor\t\t",data)
        data.x = None
        data.edge_attr = None
        data.y = None
        
        #%% add masks
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.val_mask = val_mask
        
        # print("after\t\t",data)
        
        #%% Execute Node2Vec to get node embeddings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=40,
                     context_size=20, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
        
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        
        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)
        
        
        @torch.no_grad()
        def test():
            model.eval()
            z = model()
            acc = model.test(z[data.train_mask], data.y[data.train_mask],
                             z[data.test_mask], data.y[data.test_mask],
                             max_iter=10)
            return acc
        
        
        for epoch in range(1, 201):
            loss = train()
            #acc = test()
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
                
        z = model()
        
        #%%visualize node embedding
        # from tensor to numpy
        emb_128 = z.detach().cpu().numpy()

        #%%
        dist_out = 1-pairwise_distances(emb_128, metric="cosine")
        for j in range(len(emb_128)):
            dist_out[j,j]=0
        sns.heatmap(dist_out, cmap='jet',vmin=-.01,vmax=.1,ax=ax[1])
        n2v_mat.append(dist_out)
        break
        
        # plt.figure()
        # df = pd.DataFrame(emb_128.T)
        # corr_matrix = np.array(df.corr())
        # for j in range(len(emb_128)):
        #     corr_matrix[j,j]=0
        # sns.heatmap(corr_matrix, cmap='jet',vmin=-.01, vmax=.1)
    
    return n2v_mat