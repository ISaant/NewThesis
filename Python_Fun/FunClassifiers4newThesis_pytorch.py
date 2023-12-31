#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 18:44:45 2023

@author: isaac
"""

import scipy 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch_geometric.utils import add_self_loops, to_dense_adj, to_dense_batch, dense_to_sparse
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_networkx
from torch_scatter import scatter_add
import torch.nn.functional as F
from scipy import sparse
import time
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,BatchNorm,SAGEConv, dense_diff_pool, DenseSAGEConv, GINConv
from torch_geometric.nn import global_mean_pool,global_add_pool, global_max_pool
from math import ceil
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
#%% ===========================================================================
def Scale(Data):
    
    scaler=StandardScaler()
    scaler.fit(Data)
    Data=scaler.transform(Data)
    return Data

#%% ===========================================================================
def Scale_out(Data):
    
    scaler=MinMaxScaler(feature_range=(-1,1))
    scaler.fit(Data)
    Data_scaled=scaler.transform(Data)
    return Data_scaled,scaler
#%% ===========================================================================
def Split(Data,labels,testSize,seed=None):
    idx = np.arange(len(Data))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(Data, labels, idx, test_size=testSize,random_state=seed)
    return  x_train, x_test, y_train, y_test, idx_train, idx_test

#%% ===========================================================================

def Split_Parallel_NN(psd,anat,delta,theta,alpha, beta, gamma1,gamma2,labels,testSize,seed=None):
    idx = np.arange(len(psd))
    psd_train,psd_test,anat_train,anat_test,delta_train,delta_test,\
        theta_train,theta_test, alpha_train,alpha_test,\
            beta_train, beta_test, gamma1_train,gamma1_test,\
                gamma2_train,gamma2_test, y_train, y_test,\
                    idx_train, idx_test = train_test_split(psd,anat,
                                                           delta,theta,
                                                           alpha, beta,
                                                           gamma1,gamma2,
                                                           labels, idx,
                                                           test_size=testSize,
                                                           random_state=seed)
   
    return  psd_train,psd_test,anat_train,anat_test,delta_train,delta_test,\
            theta_train,theta_test, alpha_train,alpha_test,\
            beta_train, beta_test, gamma1_train,gamma1_test,\
            gamma2_train,gamma2_test, y_train, y_test,\
            idx_train, idx_test
#%% ===========================================================================
def plotPredictionsReg(predictions,y_test,plot):
    pearson=scipy.stats.pearsonr(predictions,y_test)
    if plot :
        plt.figure()
        plt.scatter(predictions,y_test)
        
        # print(pearson)
        lims=[min(y_test)-1,max(y_test)+1]
        plt.plot(lims,lims)
        plt.xlabel('predicted')
        plt.ylabel('ture values')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()
    return pearson[0]

#%% ===========================================================================

class CustomDataset(Dataset):

    def __init__(self,psd,anat, fc, psd_s200, anat_s200, ScFeat, labels,transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.n_samples = psd.shape[0]

        # here the first column is the class label, the rest are the features
        self.psd = torch.from_numpy(Scale(psd).astype('float32')) # size [n_samples, n_features]
        self.anat = torch.from_numpy(Scale(anat).astype('float32'))
        self.fc = torch.from_numpy(fc.astype('float32'))
        self.psd_s200 = torch.from_numpy(Scale(psd_s200).astype('float32'))
        self.anat_s200 = torch.from_numpy(Scale(anat_s200).astype('float32'))
        self.ScFeat = torch.from_numpy(ScFeat.astype('float32'))
        self.labels = torch.from_numpy(labels.astype('float32'))
        
        self.transform=transform
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        sample = [self.psd[idx], self.anat[idx], self.fc[idx], self.psd_s200[idx],
                  self.anat_s200[idx], self.ScFeat[idx], self.labels[idx]]
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples 

### not in use 
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
###    

    #%% Define first nn

class NeuralNet(nn.Module):
    def __init__(self,input_size, output_size):
        super(NeuralNet,self).__init__()
        self.il = nn.Linear(input_size, 512)
        self.hl1 = nn.Linear(512, 256)
        self.hl2 = nn.Linear(256, 64)
        self.hl3 = nn.Linear(64, 16)
        self.hl4 = nn.Linear(16, 16)
        self.ol = nn.Linear(16, output_size)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ELU()
        self.activation4 = nn.SELU()
        self.activation5 = nn.ReLU()
        self.activation6 = nn.Tanh()

    def forward(self, x):
        out = self.activation1(self.il(x))
        out = self.activation2(self.hl1(out))
        out = self.activation3(self.hl2(out))
        out = self.activation4(self.hl3(out))
        out = self.activation5(self.hl4(out))
        out = self.ol(out)
        return out

#%% Define nn for graph embeddings

class NeuralNet4Graph_emb(nn.Module):
    def __init__(self,input_size, output_size):
        super(NeuralNet4Graph_emb,self).__init__()
        self.il = nn.Linear(input_size, 512)
        self.hl1 = nn.Linear(512, 256)
        self.hl2 = nn.Linear(256, 64)
        self.hl3 = nn.Linear(64, 16)
        self.hl4 = nn.Linear(16, 16)
        self.ol = nn.Linear(16, output_size)
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.dropout3 = torch.nn.Dropout(p=0.3)
        self.dropout4 = torch.nn.Dropout(p=0.3)
        self.dropout5 = torch.nn.Dropout(p=0.3)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ELU()
        self.activation4 = nn.SELU()
        self.activation5 = nn.Tanh()

    def forward(self, x):
        out = self.activation1(self.il(x))
        # out = self.dropout1(out)
        out = self.activation2(self.hl1(out))
        # out = self.dropout2(out)
        out = self.activation3(self.hl2(out))
        # out = self.dropout3(out)
        out = self.activation4(self.hl3(out))
        # out = self.dropout4(out)
        out = self.activation5(self.hl4(out))
        # out = self.dropout5(out)
        out = self.ol(out)
        return out

#%%

class CustomModel_NoFc(nn.Module):
    def __init__(self, input_size_psd, input_size_anat, output_size):
        super(CustomModel_NoFc, self).__init__()

        # Definir las capas para InputPSD
        self.NN0 = nn.Sequential(
            nn.Linear(input_size_psd, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.SELU(),
            nn.Linear(16, 16),
            nn.GELU()
        )

        # Definir las capas para InputAnat
        self.NN1 = nn.Sequential(
            nn.Linear(input_size_anat, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.SELU(),
            nn.Linear(16, 16),
            nn.GELU()
        )


        # Capa de salida
        self.output_layer = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, inputs):
        input_psd = inputs[0]
        input_anat = inputs[1]
        # Propagación a través de las capas correspondientes a InputPSD
        out0 = self.NN0(input_psd)

        # Propagación a través de las capas correspondientes a InputAnat
        out1 = self.NN1(input_anat)

        # Concatenar las salidas
        concatenated_output = torch.cat((out0, out1), dim=1)

        # Propagación a través de la capa de salida
        final_output = self.output_layer(concatenated_output)

        return final_output

#%%
class CustomModel_Sc(nn.Module):
    def __init__(self, input_size_psd, input_size_anat, input_size_ScFeat, output_size):
        super(CustomModel_Sc, self).__init__()

        # Definir las capas para InputPSD
        self.NN0 = nn.Sequential(
            nn.Linear(input_size_psd, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.SELU(),
            nn.Linear(16, 16),
            nn.GELU()
        )

        # Definir las capas para InputAnat
        self.NN1 = nn.Sequential(
            nn.Linear(input_size_anat, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.SELU(),
            nn.Linear(16, 16),
            nn.GELU()
        )

        self.NN2 = nn.Sequential(
            nn.Linear(input_size_ScFeat, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 16),
            nn.SELU(),
            nn.Linear(16, 16),
            nn.GELU()
        )

        # Capa de salida
        self.output_layer = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, inputs):
        input_psd = inputs[0]
        input_anat = inputs[1]
        input_FeatSc = inputs[2]
        # Propagación a través de las capas correspondientes a InputPSD
        out0 = self.NN0(input_psd)

        # Propagación a través de las capas correspondientes a InputAnat
        out1 = self.NN1(input_anat)

        # Propagación a través de las capas correspondientes a SC
        out2 = self.NN2(input_FeatSc)

        # Concatenar las salidas
        concatenated_output = torch.cat((out0, out1,out2), dim=1)

        # Propagación a través de la capa de salida
        final_output = self.output_layer(concatenated_output)

        return final_output

#%% To reset wights
def weight_reset(m):
    # print('yes')
    if hasattr(m, 'reset_parameters'):
        # print(m)
        m.reset_parameters()


#%% Create Dataset for graph regression
from torch_geometric.data import Data, Batch
import networkx as nx
def Dataset_graph(features, labels, connectome, task, idx=None):
    features = np.array(features,dtype=float)
    data_list=[]
    le = LabelEncoder()
    encoded=le.fit_transform(labels)
    # task=torch.FloatTensor(task)
    if type(connectome) == list:
        for sub in range(len(connectome)):
            connectome[sub] = connectome[sub].astype('float32')
    else:
        connectome=connectome.astype('float32')
    task=torch.FloatTensor(task[:,np.newaxis,np.newaxis])
    # task=torch.FloatTensor(task[:,np.newaxis,])

    for i in range(len(features)):
        x=features[i,:,:].T
        if idx:
            x=np.delete(features[i,:,:],idx[i],axis=1).T
        x=torch.FloatTensor(x)
        if type(connectome) == list:
            edge_index,edge_wight=from_scipy_sparse_matrix(sparse.csr_matrix(connectome[i]))
        else:
            edge_index,edge_wight=from_scipy_sparse_matrix(sparse.csr_matrix(connectome[i,:,:]))

        edge_index=torch.from_numpy(edge_index.numpy())
        # print(edge_index.shape)
        edge_wight=torch.from_numpy(edge_wight.numpy()).float()
        # t=task[i][np.newaxis,]
        data=Data(x=x,edge_index=edge_index,edge_attr=edge_wight,y=task[i])
        data_list.append(data)
    # print(data)
    dataset=Batch().from_data_list(data_list)
    return dataset

#%% Graph neural net w/o pull

class GCN(torch.nn.Module):
    def __init__(self, num_node_features,  hidden_channels, lin = True):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.3)
        
        if lin is True:
            self.lin = torch.nn.Linear(2*hidden_channels,
                                       hidden_channels)
        else:
            self.lin = None
        
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)  
        self.il = nn.Linear(hidden_channels, 64)
        self.hl1 = nn.Linear(64, 16)
        self.hl2 = nn.Linear(16, 16)
        self.ol = nn.Linear(16, 1)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        # self.pooling=nn.AdaptiveAvgPool2d((hidden_channels,2))
        self.activation3 = nn.ELU()
        # self.activation4 = nn.SELU()
        # self.activation5 = nn.GELU()

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index
        batch=data.batch
        edge_attr=data.edge_attr
        # 1. Obtain node embeddings 
        x1 = self.bn1(self.conv1(x, edge_index, edge_attr))
        x1 = self.activation1(x1)
        x1 = self.dropout(x1)
        x2 = self.bn2(self.conv2(x1, edge_index,edge_attr)).relu()
        x2 = self.dropout(x2)
        x = torch.cat([x1, x2], dim=-1)
        # x = self.conv3(x, edge_index)
        
         
        if self.lin is not None:
            x = self.lin(x).relu()
        # else:
        #     x=x

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # x= self.pooling(x)
        # print(x.shape)
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        out = self.activation1(self.il(x))
        out = self.activation2(self.hl1(out))
        out = self.activation3(self.hl2(out))
        out = self.ol(out)
        
        return out
    
#%%

class SAGE_GCN(torch.nn.Module):
    def __init__(self, num_node_features,  hidden_channels, lin=True):
        super(SAGE_GCN, self).__init__()
        
        self.lin1 = Linear(num_node_features, 512)
        self.lin2 = Linear(512, 64)
        # self.lin3 = Linear(64, 64)
        
        
        self.conv1 = SAGEConv(64, hidden_channels, aggr=['mean', 'max','sum','std', 'var'],normalize=True)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=['mean', 'max','sum','std', 'var'],normalize=True)
        self.bn2 = BatchNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.3)
        
        if lin is True:
            self.lin = torch.nn.Linear(2*hidden_channels,
                                       hidden_channels)
        else:
            self.lin = None
        
        self.il = Linear(hidden_channels, 64)

        self.hl1 = Linear(64, 16)
        self.hl2 = Linear(16, 16)
        self.ol = Linear(16, 1)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        # self.pooling=nn.AdaptiveAvgPool2d((hidden_channels,2))
        # self.activation3 = nn.ELU()
        # self.activation4 = nn.SELU()
        # self.activation5 = nn.GELU()

    def forward(self, data):
        # 1. Obtain node embeddings 
        x=data.x
        edge_index=data.edge_index
        batch=data.batch
        # edge_index=add_self_loops(edge_index)

        x1 = self.lin1(x)
        x1= F.sigmoid(x1)
        x1 = self.lin2(x1)
        x1= F.relu(x1)
        # x = self.lin3(x)
        # x= F.relu(x)
        x1 = self.bn1(self.conv1(x1, edge_index))
        x1 = self.dropout(x1)
        x1 = F.relu(x1)
        # x, edge_index, batch, perm, score = self.pool1(x, edge_index, batch)
        
        x2 = self.bn2(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        x2 = F.relu(x2)
        # x, edge_index, batch, perm, score = self.pool2(x, edge_index, batch)
        x = torch.cat([x1, x2], dim=-1)
        # print(x.shape)

        # x = self.bn3(self.conv3(x, edge_index))
        x = self.dropout(x)
        # x = F.elu(x)
        # x, edge_index, batch, perm, score = self.pool3(x, edge_index, batch)
        
        if self.lin is not None:
            x = self.lin(x).relu()
        else:
            x=x2
        
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # print(x.shape)
        # x= x.view(-1,x.shape[0]*x.shape[1])
        # print(x.shape)
        # x= self.pooling(x)
        # print(x.shape)
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        out = self.activation1(self.il(x))
        out = self.activation2(self.hl1(out))
        out = self.activation2(self.hl2(out))
        out = self.ol(out)
        
        return out


#%% 
max_nodes=68
class GNN_emb(torch.nn.Module):
    def __init__(self, num_node_features,  hidden_channels, out_channels, normalize=False, lin=True):
        super(GNN_emb,self).__init__()
        
        self.conv1 = DenseSAGEConv( num_node_features, hidden_channels,normalize=True)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels,normalize=True)
        self.bn2 = BatchNorm1d(out_channels)
        self.activation1 = nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.3)
        if lin is True:
            self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None
            
    def bn(self, i, x): #Normalization
        batch_size, num_nodes, num_channels = x.size()
        # print('----')
        # print(x.size())
        x = x.view(-1, num_channels)
        # print(x.size())
        x = getattr(self, f'bn{i}')(x)
        # print(x.size())
        x = x.view(batch_size, num_nodes, num_channels)
        # print(x.size())
        return x
        
    def forward(self, x, adj, mask=None):
        # 1. Obtain node embeddings 
        x1 = self.bn(1,self.conv1(x, adj, mask))
        x1 = self.activation1(x1)
        x1 = self.dropout(x1)
        x2 = self.bn(2,self.conv2(x1, adj, mask)).relu()
        x2 = self.dropout(x2)
        x = torch.cat([x1, x2], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x
    
    
class GNN_DiffPool(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN_DiffPool, self).__init__()
        self.lin1_pre = Linear(num_node_features, 512)
        self.lin2_pre = Linear(512, num_node_features)
        
        num_nodes = ceil(0.5 * max_nodes)
        self.gnn1_pool = GNN_emb(num_node_features, 64, num_nodes)
        self.gnn1_embed = GNN_emb(num_node_features, 64, 64, lin=False)

        self.num_nodes = ceil(0.15 * num_nodes)
        self.gnn2_pool = GNN_emb(2 * 64, 64, self.num_nodes)
        self.gnn2_embed = GNN_emb(2 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN_emb(2 * 64, 64, 64, lin=False)

        self.lin1_post = torch.nn.Linear(2 * 64 *self.num_nodes, 64)
        # self.lin1_post = torch.nn.Linear(2 * 64, 64)
        self.lin2_post = torch.nn.Linear(64, 16)
        self.lin3_post = torch.nn.Linear(16, 16)
        self.ol = Linear(16, 1)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()

    def forward(self, data):
        
        x=data.x
        edge_index=data.edge_index
        batch=data.batch
        # edge_index=add_self_loops(edge_index)

        x = self.lin1_pre(x)
        x= F.sigmoid(x)
        x = self.lin2_pre(x)
        x= F.relu(x)
        
        out,mask=to_dense_batch(x=x,batch=batch)
        adj= to_dense_adj(edge_index=edge_index,batch=batch)
        s = self.gnn1_pool(out, adj, mask)
        x = self.gnn1_embed(out, adj, mask)
        # print (s.shape,x.shape)
        # x = x[None, :]
        # s = s[None, :]
        
        # print (s.shape,x.shape)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        # print(x.shape)
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        
        
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj)
        # print(x.shape)
        x = x.view(-1,2 * 64 *self.num_nodes)
        # x = x.mean(dim=1)
        out = self.activation1(self.lin1_post(x))
        out = self.activation2(self.lin2_post(out))
        out = self.activation2(self.lin3_post(out))
        out = self.ol(out)
        return out


#%% 

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP,self).__init__()
        self.il = nn.Linear(input_size, 512)
        self.hl1 = nn.Linear(512, 256)
        self.hl2 = nn.Linear(256, 64)
        self.hl3 = nn.Linear(64, 64)
        self.ol = nn.Linear(64, output_size)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ELU()
        self.activation4 = nn.SELU()

    def forward(self, x):
        out = self.activation1(self.il(x))
        out = self.activation2(self.hl1(out))
        out = self.activation3(self.hl2(out))
        out = self.activation4(self.hl3(out))
        out = self.ol(out)
        return out

class GNN_GIN(torch.nn.Module):
    def __init__(self, input_size, train_eps, output_size=32):
        super(GNN_GIN,self).__init__()
        self. output_size=output_size
        self.lin1_pre = Linear(input_size, 512)
        self.lin2_pre = Linear(512, input_size)
        
        mlp1=MLP(input_size, output_size)
        self.conv1 = GINConv(mlp1,train_eps=train_eps)
        self.bn1 = BatchNorm(output_size)
        
        mlp2=MLP(output_size, output_size)
        self.conv2=GINConv(mlp2,train_eps=train_eps)
        self.bn2=BatchNorm(output_size)
        
        mlp3=MLP(output_size, output_size)
        self.conv3=GINConv(mlp3,train_eps=train_eps)
        self.bn3=BatchNorm(output_size)
        
        self.dropout = torch.nn.Dropout(p=0.3)
        
        self.il = nn.Linear(3*output_size, 512)
        self.hl1 = nn.Linear(512, 256)
        self.hl2 = nn.Linear(256, 64)
        self.hl3 = nn.Linear(64, 16)
        self.hl4 = nn.Linear(16, 16)
        self.ol = nn.Linear(16, 1)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ELU()
        self.activation4 = nn.SELU()
        self.activation5 = nn.GELU()
        
    def forward(self,data):
        x=data.x
        edge_index=data.edge_index
        batch = data.batch
        # edge_index=add_self_loops(edge_index)
        # print(edge_index.shape, edge_index.dtype)
        x = self.lin1_pre(x)
        x= F.sigmoid(x)
        x = self.lin2_pre(x)
        x= F.relu(x)
        x1 = F.sigmoid(self.bn1(self.conv1(x, edge_index)))
        # x1 = self.dropout(x1)
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        # x2 = self.dropout(x2)
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        # x3 = self.dropout(x3)
        
        x = torch.cat([x1, x2, x3], dim=-1)
        # print (x.shape)
        # x = x.view(-1,3 * self. output_size *num_nodes)
        x = global_add_pool(x,batch)
        # print (x.shape)
        out = self.activation1(self.il(x))
        out = self.activation2(self.hl1(out))
        out = self.activation3(self.hl2(out))
        out = self.activation4(self.hl3(out))
        out = self.activation5(self.hl4(out))
        out = self.ol(out)
        return out
#%%
def train_ANN(model, criterion,optimizer, train, val, epochs, var_pos):
    n_total_steps = len(train)  # number of batches
    for epoch in range(epochs):
        for i, data in enumerate(train):
            targets = data[-1].to(device)
            if type(var_pos) == int:
                var = data[var_pos].to(device)
                outputs = model(var)
            else:
                varss=[]
                for pos in var_pos:
                    varss.append(data[pos].to(device))
                outputs = model(varss)
            loss = criterion(outputs, targets)

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if epoch % 10 == 0:
            #     mse_val, _ = test_ANN(model, val, var_pos)
            #     mse_train, _ = test_ANN(model, train, var_pos)
            #     print(f'Epoch: {epoch:02d}, : MSE_train: {mse_train:.4f}, MSE_val: {mse_val:.4F}')
    outputs= outputs.detach().to('cpu').numpy().flatten()
    targets= targets.detach().to('cpu').numpy().flatten()
    mse_train, _ = test_ANN(model, train, var_pos)
    NNPred = plotPredictionsReg(outputs,targets, False)
    print(f'Train_Acc: {NNPred:4f}, Train_MSE: {mse_train}')
# testing and eval

def test_ANN(model, loader, var_pos):
    model.eval()
    pred = []
    with torch.no_grad():
        for data in loader:
            targets = data[-1].to('cpu').numpy().flatten()
            if type(var_pos) == int:
                var = data[var_pos].to(device)
                outputs = model(var).to('cpu').numpy().flatten()
            else:
                varss = []
                for pos in var_pos:
                    varss.append(data[pos].to(device))
                outputs = model(varss).to('cpu').numpy().flatten()
            mse = mean_squared_error(targets,outputs)
            pred.extend(outputs)
    return mse, pred

#%%

def train_GNN(model, criterion,optimizer, train, val, epochs):
    n_total_steps = len(train)  # number of batches
    for epoch in range(epochs+1):
        for i, data in enumerate(train):
            data = data.to(device)
            out = model(data)  # Perform a single forward pass. Intenta agregar edge_attr
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()
            targets = data.y.to('cpu').numpy()
            if epoch % 10 == 0 and i == 0:
                mse_val, _, _ = test_GNN(model, val, False)
                mse_train, _, _ = test_GNN(model, train, False)
                print(f'Epoch: {epoch:02d}, : MSE_train: {mse_train:.4f}, MSE_val: {mse_val:.4F}')
    mse_train, _, Acc = test_GNN(model, train, False)
    print(f'Train_Acc: {Acc:4f}, Train_MSE: {mse_train}')
# testing and evalS

def test_GNN(model, loader, plot):
    pred = []
    true_label = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).to('cpu').numpy()
            pred.extend(out)
            true_label.extend(data.y.to('cpu').numpy())
        pred = np.array(pred)
        true_label = np.array(true_label)
        mse = mean_squared_error(np.array(true_label),pred)
        NNPred=plotPredictionsReg(pred.flatten(),true_label.flatten(),plot)# Check against ground-truth labels.

    return mse, pred ,NNPred

#%%
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline


def svm_reg(connectome, y, kernel):
    vectorize_tril=[]
    for adj in connectome:
        lower_triangle=np.tril(adj, -1).flatten()
        lower_triangle=lower_triangle[np.where(lower_triangle)]
        vectorize_tril.append(lower_triangle)

    vectorize_tril=np.array(vectorize_tril)
    X_train, X_test, y_train, y_test= train_test_split(vectorize_tril, y, test_size=.3, random_state=0)

    y_train = y_train.reshape(-1, 1)
    X_sc = StandardScaler()
    y_sc = StandardScaler()
    X_train = X_sc.fit_transform(X_train)
    y_train = y_sc.fit_transform(y_train)
    y_train = y_train.flatten()
    regrassor = SVR(kernel=kernel, C=1.5, epsilon=0.2)
    regrassor.fit(X_train, y_train)
    pred = regrassor.predict(X_sc.transform(X_test))
    pred = pred.reshape(-1, 1)
    pred = y_sc.inverse_transform(pred).flatten()
    true_label = y_test.flatten()
    NNPred = plotPredictionsReg(pred, true_label, True)  # Check against ground-truth labels.
    print(NNPred)