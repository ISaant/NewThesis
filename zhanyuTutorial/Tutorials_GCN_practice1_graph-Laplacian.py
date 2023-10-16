#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:37:34 2023

@author: isaac
"""

import os
import sys
import math
import time
import datetime

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse
from scipy.stats import spearmanr
from sklearn import preprocessing, metrics,manifold
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from nilearn import connectome
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch_sparse import spmm
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops

from utils import load_fmri_data_from_lmdb, extract_event_data, HCP_taskfmri_matrix_datasets, fmri_samples_collate_fn

import warnings
warnings.filterwarnings(action='once')

#%% step1: load brain connectome
import scipy.io
import matplotlib.pyplot as plt

adjacent_mat_file = "MMP_DiffusionConnectivity_HCP_avg56.mat"
mat = scipy.io.loadmat(adjacent_mat_file)
corr_matrix_z = mat['SC_avg56']
num_nodes = corr_matrix_z.shape[0]

fig=plt.figure(figsize=(10,4))
plt.imshow(corr_matrix_z, cmap="jet")
plt.colorbar()

fig=plt.figure(figsize=(10,4))
plt.hist(corr_matrix_z[corr_matrix_z<1])

#%% Step2: build a k-NN graph.

def build_adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M,k == idx.shape # comprobaciones para asegurarse de que las dimensiones sean correctas y de que las distancias sean no negativas.
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2 #Calcula el cuadrado de la media de las distancias a los vecinos más cercanos
    dist = np.exp(- dist**2 / sigma2) # Pasamos de correlacion a distancia, entre mayor la correlacion menor la distancia

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = sparse.coo_matrix((V, (I, J)), shape=(M, M)) #Crea una matriz de 
    #coordenadas dispersas (sparse). Es importante para el calculo de los eigenvalues y eigenvectors 
    # No self-connections.
    W.setdiag(0)

    # Non-directed graph. Asegura que el grafo sea no dirigido al comparar las
    #entradas de la matriz W con sus transpuestas y tomando el mínimo entre 
    #los dos valores. Esto garantiza que las conexiones sean bidireccionales.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W

Nneighbours=8
idx = np.argsort(-corr_matrix_z)[:, 0:Nneighbours + 1]#se buscan los k valores mas grandes por fila, se busca a partir de la segunda columns porque se le hincharon sus huevos  
dist = np.array([corr_matrix_z[i, idx[i]] for i in range(corr_matrix_z.shape[0])])
dist[dist < 0.1] = 0
adj_mat_sp = build_adjacency(dist, idx)

fig=plt.figure(figsize=(10,4))
fig.add_subplot(121)
plt.imshow(adj_mat_sp.todense(), cmap="jet",vmin=0.1, vmax=0.5);
plt.colorbar()

fig.add_subplot(122)
plt.hist(adj_mat_sp.data)

#%% Step 3: Laplacian Matrix: L=I-D(-1/2)AD(-1/2)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # Esta función convierte una matriz dispersa (sparse matrix) en formato 
    # scipy.sparse a un tensor disperso (sparse tensor) de PyTorch.
    # La conversión implica la creación de un tensor para los índices 
    # de la matriz dispersa y otro tensor para los valores de la matriz.
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat_sp)
print("Converting from scipy sparse matrix:")
print(adj_mat_sp.indices)
print(adj_mat_sp.data)
print("Converting to torch sparse tensor:")
print(adj_mat._indices())
print(adj_mat._values())

# se extraen los índices de las conexiones del grafo en edge_index y 
# los pesos de los bordes en edge_weight. Estos son los componentes 
# necesarios para calcular la matriz Laplaciana del grafo.
edge_index = adj_mat._indices()
edge_weight = adj_mat._values()
row, col = edge_index
        
#degree-matrix
# Se calcula la matriz de grados del grafo (deg). La matriz de grados contiene
# la suma de los pesos de las aristas conectados a cada nodo. Esta información
# es fundamental para calcular la matriz Laplaciana normalizada.
deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

# Compute normalized and rescaled Laplacian.
deg = deg.pow(-0.5)
deg[torch.isinf(deg)] = 0
lap = deg[row] * edge_weight * deg[col]

###Rescale the Laplacian eigenvalues in [-1, 1]
##rescale: 2L/lmax-I; lmax=1.0
# En esta parte, se realiza una operación de reescalado de los valores propios 
# de la matriz Laplaciana. La matriz resultante se parece a 2L/lmax - I, donde 
# lmax se establece en 1.0. Esta operación de reescalado es común en análisis
# espectral de grafos y puede ayudar a ajustar los valores propios en un rango deseado

# Aquí, se agrega un bucle a cada nodo del grafo, que se conoce como 
# "self-loop". El valor del bucle se establece en fill_value, que es 1 en 
# este caso. Esto se hace para asegurarse de que cada nodo tenga una 
# conexión consigo mismo en la matriz Laplaciana.
fill_value = 1  ##-0.5
edge_index, lap = add_self_loops(edge_index, -lap, fill_value, num_nodes)

laplacian_matrix = sparse.coo_matrix((lap.numpy(),edge_index),shape=(num_nodes,num_nodes))
fig=plt.figure(figsize=(10,4))
fig.add_subplot(121)
plt.imshow(-laplacian_matrix.todense(), cmap="jet",vmin=0.01, vmax=0.5);
plt.colorbar()

fig.add_subplot(122)
plt.hist(-laplacian_matrix.data)

#%% Step 4: Spectral decomposition
from scipy.linalg import eigh
w, v = eigh(laplacian_matrix.todense()) # w = eigenvalues, v = eigenvectors

K_eigbasis = min(4,num_nodes)
fig=plt.figure(figsize=(15,4))
fig.add_subplot(131)
plt.scatter(range(K_eigbasis),w[:K_eigbasis])
plt.title("Fist {} EigenValues".format(K_eigbasis))

fig.add_subplot(132)
plt.imshow(np.repeat(v[:,:K_eigbasis],100,axis=1), cmap="Spectral")
# plt.imshow(v[:,:K_eigbasis],aspect='auto', cmap="Spectral")
plt.title("Fist {} EigenVectors".format(K_eigbasis))

fig.add_subplot(133)
plt.plot(v[:,1])
plt.title("2nd EigenVectors")

#%% Example #1: mapping 2nd and 3rd eigvectors and spectral clustering

##spectral clustering from sklearn
from sklearn import cluster
sp_clustering = cluster.SpectralClustering(n_clusters=3,eigen_solver='arpack',affinity='precomputed',
                                    assign_labels="discretize",random_state=1234).fit(corr_matrix_z)
sp_clustering.labels_ #This is better!

##spectral clustering: applying k-means to first k eigenvectors
sk_clustering = cluster.KMeans(n_clusters=3, random_state=1234).fit(v[:,:K_eigbasis])
sk_clustering.labels_
clusters = np.array([sp_clustering.labels_,sk_clustering.labels_]).transpose()
print(np.corrcoef(sp_clustering.labels_,sk_clustering.labels_))

fig=plt.figure(figsize=(15,4))
fig.add_subplot(131)
plt.plot(clusters)

##visualize the graph architecture
import networkx as nx
graph = nx.from_scipy_sparse_array(adj_mat_sp, parallel_edges=False, create_using=None, edge_attribute='weight')
pos = nx.spring_layout(graph) 
fig.add_subplot(132)
nx.draw(graph, pos, node_size=20, node_color='b')

pos = {i : (v[i,1], v[i,2]) for i in range(num_nodes)}
fig.add_subplot(133)
# set of nodes
nx.draw(graph, pos, node_size=20, node_color='b')
nx.draw_networkx_nodes(graph, pos, 
                        nodelist=list(np.where(sp_clustering.labels_==1)[0]),
                        node_color='r',node_size=20)
nx.draw_networkx_nodes(graph, pos, 
                        nodelist=list(np.where(sp_clustering.labels_==0)[0]),
                        node_color='b',node_size=20,)

nx.draw_networkx_nodes(graph, pos, 
                        nodelist=list(np.where(sp_clustering.labels_==2)[0]),
                        node_color='g',node_size=20,)

plt.show()

#%% Example #2: brain decoding using graph convolutional networks

#step1: load fmri data and event labels

fmri_file = "MOTOR_MMP_ROI_act_test_sub100.lmdb"
event_file = "MOTOR_event_labels_test_sub100_newLR.h5"

sub100_tc_matrix, subname100 = load_fmri_data_from_lmdb(fmri_file)
sub100_trial_labels, sub100_trialID, sub100_sub_name, sub100_coding_direct = extract_event_data(event_file)
print(np.array(sub100_tc_matrix).shape,np.array(sub100_trial_labels).shape)

Subject_Num = np.array(sub100_tc_matrix).shape[0]
Trial_Num = np.array(sub100_tc_matrix).shape[1]
TR = 0.72

tc_roi_matrix = sub100_tc_matrix[0].transpose()
event_design = sub100_trial_labels.iloc[0]
event_select = np.zeros((Trial_Num,1))
for ei in np.where(event_design!='rest'):
    event_select[ei]=1

##plot regional tc aligned with event design
region_index = np.argsort(tc_roi_matrix.var(axis=-1))
tc_roi_matrix_sort = [tc_roi_matrix[ii,:] for ii in region_index]
event_time = np.arange(Trial_Num)*TR

fig=plt.figure(figsize=(10,4))
fig.add_subplot(121)
plt.imshow(tc_roi_matrix_sort)

fig.add_subplot(122)
fmri_tc = preprocessing.MinMaxScaler().fit_transform(tc_roi_matrix[region_index[350],:].reshape(-1,1))
plt.plot(event_time,fmri_tc,'b')
plt.plot(event_time, event_select,'r--')

#%% build the label and data matrix
#print(task_contrasts)
task_contrasts = ['footL_mot','footR_mot','handL_mot','handR_mot','tongue_mot']
le = preprocessing.LabelEncoder()
le.fit(task_contrasts)
label_trial_task = np.array(le.transform(event_design[event_design!='rest']))
label_trial_task = np.array(np.split(label_trial_task, np.where(np.diff(label_trial_task))[0] + 1))[:,0] # se busca el cuando cambia de tarea 
task_idx = np.argsort(label_trial_task)
task_names = [le.inverse_transform(label_trial_task)[idx] for idx in task_idx]

###extract BOLD signals according to trial info
event_trial_block = np.split(event_select.astype(int), np.where(np.diff(event_select.astype(int),axis=0))[0] + 1)
fmri_trial_block = np.split(tc_roi_matrix, np.where(np.diff(event_select.astype(int),axis=0))[0] + 1,axis=-1)

#Aqui saca el promedio de la señal para cada tarea y luego las concatena , osea que la serie de tiempo se vuelve un escalar por region
fmri_trial_task = []
for ii in range(len(event_trial_block)):
    if np.unique(event_trial_block[ii]) !=0 :
        ##print(fmri_trial_block[ii].shape)
        fmri_trial_task.append(fmri_trial_block[ii].mean(axis=-1))
fmri_trial_task = np.array(fmri_trial_task)
print(fmri_trial_task.shape)
fig=plt.figure(figsize=(10,4))


correlation = np.corrcoef([fmri_trial_task[idx,:] for idx in task_idx])
plt.imshow(correlation, cmap="jet",vmin=0.0, vmax=1.0)
plt.yticks(np.arange(10), task_names)
plt.colorbar()
            
# ###map BOLD signals to 2d space  
# # tsne = manifold.TSNE(n_components=2, init='pca', random_state=1234, n_iter=10000)
# # output_2d = tsne.fit_transform(fmri_trial_task)
# # fig.add_subplot(122)
# # plt.scatter(output_2d[:, 0], output_2d[:, 1], c=label_trial_task)
# # plt.tight_layout()
# # plt.show()

low_mode = v[:,0]
low_mode = np.matmul(tc_roi_matrix.T,low_mode)
high_mode = v[:,-1]
high_mode = np.matmul(tc_roi_matrix.T,high_mode)

fig=plt.figure(figsize=(14,6))
fig.add_subplot(121)
fmri_tc = preprocessing.MinMaxScaler().fit_transform(low_mode.reshape(-1,1))
plt.plot(event_time,fmri_tc,'b')
plt.plot(event_time, event_select,'r--')
plt.title('Low frequency graph mode of fmri tc')

fig.add_subplot(122)
fmri_tc = preprocessing.MinMaxScaler().fit_transform(high_mode.reshape(-1,1))
plt.plot(event_time,fmri_tc,'b')
plt.plot(event_time, event_select,'r--')
plt.title('High frequency graph mode of fmri tc')

#%% nonlinear mapping
kmodes=20
laplacian_mode1 = np.matmul(tc_roi_matrix.T,v[:,1:kmodes+1]).transpose() #Esta asociando una serie de tiempo con cada eigenvector extraido del la matriz laplaciana
laplacian_mode2 = np.matmul(tc_roi_matrix.T,v[:,100:100+kmodes]).transpose()
laplacian_mode3 = np.matmul(tc_roi_matrix.T,v[:,kmodes:]).transpose()

print(laplacian_mode1.shape)
###extract BOLD signals according to trial info
event_trial_block = np.split(event_select.astype(int), np.where(np.diff(event_select.astype(int),axis=0))[0] + 1)
fmri_trial_block = np.split(laplacian_mode1, np.where(np.diff(event_select.astype(int),axis=0))[0] + 1,axis=-1)
fmri_trial_task = []
for ii in range(len(event_trial_block)):
    if np.unique(event_trial_block[ii]) !=0 :
        print(ii)
        fmri_trial_task.append(fmri_trial_block[ii].flatten())
fmri_trial_task = np.array(fmri_trial_task)
print(fmri_trial_task.shape)

fig=plt.figure(figsize=(17,4))
fig.add_subplot(131)
correlation = np.corrcoef([fmri_trial_task[idx,:] for idx in task_idx])
plt.imshow(correlation, cmap="jet",vmin=0.2, vmax=1.0)
plt.yticks(np.arange(10), task_names)
plt.colorbar()
plt.title("Low freq graph mode")

fmri_trial_block = np.split(laplacian_mode2, np.where(np.diff(event_select.astype(int),axis=0))[0] + 1,axis=-1)
fmri_trial_task = []
for ii in range(len(event_trial_block)):
    if np.unique(event_trial_block[ii]) !=0 :
        fmri_trial_task.append(fmri_trial_block[ii].flatten())
fmri_trial_task = np.array(fmri_trial_task)
fig.add_subplot(132)
correlation = np.corrcoef([fmri_trial_task[idx,:] for idx in task_idx])
plt.imshow(correlation, cmap="jet",vmin=0.2, vmax=1.0)
plt.yticks(np.arange(10), task_names)
plt.colorbar()
plt.title("Mid freq graph mode")

fmri_trial_block = np.split(laplacian_mode3, np.where(np.diff(event_select.astype(int),axis=0))[0] + 1,axis=-1)
fmri_trial_task = []
for ii in range(len(event_trial_block)):
    if np.unique(event_trial_block[ii]) !=0 :
        fmri_trial_task.append(fmri_trial_block[ii].flatten())
fmri_trial_task = np.array(fmri_trial_task)
fig.add_subplot(133)
correlation = np.corrcoef([fmri_trial_task[idx,:] for idx in task_idx])
plt.imshow(correlation, cmap="jet",vmin=0.2, vmax=1.0)
plt.yticks(np.arange(10), task_names)
plt.colorbar()
plt.title("High freq graph mode")

#%% PART 2

modality='MOTOR'
task_contrasts = {"rf": "footR_mot",
                  "lf": "footL_mot",
                  "rh": "handR_mot",
                  "lh": "handL_mot",
                  "t": "tongue_mot"}
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 1}

target_name = (list(task_contrasts.values()))
print(target_name)
Nlabels = len(target_name) + 1
fmri_train_dataset = HCP_taskfmri_matrix_datasets(sub100_tc_matrix, sub100_trial_labels, target_name, block_dura=17, isTrain='train') #dim 0 are the tasks
                
train_loader = DataLoader(fmri_train_dataset, collate_fn=fmri_samples_collate_fn, **params)
i=0
for data,label in train_loader:
   i+=1 
   print(i,data.shape,label.shape)

#%% PART 3

###split the entire dataset into train and test tests
###############################
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 2}

Region_Num = sub100_tc_matrix[0].shape[-1]
block_dura = 17    
test_size = 0.2
randomseed=1234

test_sub_num = len(sub100_tc_matrix)
rs = np.random.RandomState(randomseed)
train_sid, test_sid = train_test_split(range(test_sub_num), test_size=test_size, random_state=rs, shuffle=True)
print('training on %d subjects, validating on %d subjects' % (len(train_sid), len(test_sid)))

####train set
fmri_data_train = [sub100_tc_matrix[i] for i in train_sid]
label_data_train = pd.DataFrame(np.array([sub100_trial_labels.iloc[i] for i in train_sid]))
fmri_train_dataset = HCP_taskfmri_matrix_datasets(fmri_data_train, label_data_train, target_name, block_dura=17, isTrain='train')
train_loader = DataLoader(fmri_train_dataset, collate_fn=fmri_samples_collate_fn, **params)

####test set
fmri_data_test = [sub100_tc_matrix[i] for i in test_sid]
label_data_test = pd.DataFrame(np.array([sub100_trial_labels.iloc[i] for i in test_sid]))
fmri_test_dataset = HCP_taskfmri_matrix_datasets(fmri_data_test, label_data_test, target_name, block_dura=17, isTrain='test')
test_loader = DataLoader(fmri_test_dataset, collate_fn=fmri_samples_collate_fn, **params)


#%% Finally, build our DNN model and start training
# we first test on a simple MLP with 2 hidden layer
from model import FCN
from model import count_parameters, model_fit_evaluate

###fully-connected
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

model = FCN(Region_Num, Nlabels)
model = model.to(device)
print(model)
print("{} paramters to be trained in the model\n".format(count_parameters(model)))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()
num_epochs=10

model_fit_evaluate(model,adj_mat,device,train_loader,test_loader,optimizer,loss_func,num_epochs)

#%%
#Next, we tried with graph convolutional networks
#there are two versions we test here: 1stGCN and ChebyNet;

#1stGCN only takes the first order neighborhood (direct neighbors)
#ChebyNet expands this to k-th order neighborhood (connecting by k steps)
#You can choose between these two models by changing the parameter gcn_flag from model.py

#gcn_flag=True: 1stGCN
#gcn_flag=False: ChebyNet
#By comparing the two models, we can see the benefits of including higher-order information integration

from model import ChebNet

filters=32
num_layers=2
model_test = ChebNet(block_dura, filters, Nlabels, gcn_layer=num_layers,dropout=0.25,gcn_flag=True)
#model_test = ChebNet(block_dura, filters, Nlabels, K=5,gcn_layer=num_layers,dropout=0.25)

model_test = model_test.to(device)
adj_mat = adj_mat.to(device)
print(model_test)
print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))

optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)

model_fit_evaluate(model_test,adj_mat,device,train_loader,test_loader,optimizer,loss_func,30)

#%%
model_test = ChebNet(block_dura, filters, Nlabels, K=5,gcn_layer=num_layers,dropout=0.25)

model_test = model_test.to(device)
adj_mat = adj_mat.to(device)
print(model_test)
print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))

optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)

model_fit_evaluate(model_test,adj_mat,device,train_loader,test_loader,optimizer,loss_func,30)