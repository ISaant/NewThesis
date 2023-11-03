#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:31:35 2023

@author: sflores
"""

from torch_geometric.nn import Node2Vec

import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_networkx, to_scipy_sparse_matrix
from torch_geometric.utils import to_dense_adj


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Use only one moleculas compound
data = dataset_all[1]
# extract edge list
edge_list = data.edge_index.t().numpy()
adj=to_dense_adj(data.edge_index, edge_attr=data.edge_attr)
adj=np.squeeze(adj.numpy())
plt.figure()
sns.heatmap(adj,cmap='jet')
print(edge_list[0:10])

#%%
# extract edge attributes
edge_attr = data.edge_attr.numpy()
print(edge_attr[0:10])

#%%
import networkx as nx

# build the graph
graph1 = nx.Graph()

for i in range(len(edge_list)):
    u = edge_list[i][0]
    v = edge_list[i][1]
    graph1.add_edge(u,v,label=edge_attr[i])
    
print(graph1.edges(data=True))

pos = nx.spring_layout(graph1)
nx.draw(graph1,pos)
# nx.draw_networkx_edge_labels(graph1,pos,nx.get_edge_attributes(graph1,'label'))
plt.show()


#%%
#build the graph with
#train_mask, test_mask, val_mask

np.random.seed(10)
# get the nodes
nodes = data.edge_index.t().numpy()
nodes = np.unique(list(nodes[:,0]) + list(nodes[:,1]))

np.random.shuffle(nodes) # shuffle node order
print(len(nodes))

#%%
# get train test and val sizes: (70% - 15% - 15%)
train_size = int(len(nodes)*0.9)
test_size = int(len(nodes)*0.95) - train_size
val_size = len(nodes) - train_size - test_size

# get train test and validation set of nodes
train_set = nodes[0:train_size]
test_set = nodes[train_size:train_size+test_size]
val_set = nodes[train_size+test_size:]


print(len(train_set),len(test_set),len(val_set))
print(len(train_set)+len(test_set)+len(val_set) == len(nodes))

print("train set\t",train_set[:10])
print("test set \t",test_set[:10])
print("val set  \t",val_set[:10])

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
    
print("train mask \t",train_mask[0:15])
print("test mask  \t",test_mask[0:15])
print("val mask   \t",val_mask[0:15]) 

#%% remove from the data what do we not use.

print("befor\t\t",data)
data.x = None
data.edge_attr = None
data.y = None

#%% add masks
data.train_mask = train_mask
data.test_mask = test_mask
data.val_mask = val_mask

print("after\t\t",data)

#%% Execute Node2Vec to get node embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
             context_size=10, walks_per_node=10,
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
from sklearn.decomposition import PCA
# fit and transform using PCA
pca = PCA(n_components=3)
emb2d = pca.fit_transform(emb_128)

plt.figure(figsize=(12,10))
ax=plt.subplot(projection='3d')
ax.set_title("node embedding in 2D")
ax.scatter(emb2d[:,0],emb2d[:,1],emb2d[:,2],s=200)

for i in range(len(emb2d)): 
    plt.annotate(ROIs[i], (emb2d[i,0],emb2d[i,1] + 0.1)) 
# plt.xlim((-.5, .5)) 
# plt.ylim((-.5, .5)) 
# plt.zlim((-.5, .5)) 
plt.show()

#%%
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine



dist_out = 1-pairwise_distances(emb_128, metric="cosine")
for j in range(len(emb2d)):
    dist_out[j,j]=0
sns.heatmap(dist_out, cmap='jet',vmin=-.01, vmax=.03)
