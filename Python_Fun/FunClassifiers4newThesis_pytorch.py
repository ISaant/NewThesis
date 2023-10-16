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
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_networkx
from torch_scatter import scatter_add
import torch.nn.functional as F
from scipy import sparse
import time
#%% ===========================================================================
def Scale(Data):
    
    scaler=StandardScaler()
    scaler.fit(Data)
    Data=scaler.transform(Data)
    return Data

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

    def __init__(self,psd,anat,delta_fc,theta_fc,alpha_fc,beta_fc,gamma1_fc,gamma2_fc,labels,transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.n_samples = psd.shape[0]
        # self.n2_samples = data2.shape[0]

        # here the first column is the class label, the rest are the features
        self.psd = torch.from_numpy(psd.astype('float32')) # size [n_samples, n_features]
        self.anat = torch.from_numpy(anat.astype('float32'))
        self.delta = torch.from_numpy(delta_fc)
        self.theta = torch.from_numpy(theta_fc)
        self.alpha = torch.from_numpy(alpha_fc)
        self.beta = torch.from_numpy(beta_fc)
        self.gamma1 = torch.from_numpy(gamma1_fc)
        self.gamma2 = torch.from_numpy(gamma2_fc)
        self.labels = torch.from_numpy(labels.astype('float32'))
        
        self.transform=transform
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        sample = (self.psd[idx], self.anat[idx],
                  self.delta[idx], self.theta[idx],
                  self.alpha[idx], self.beta[idx], 
                  self.gamma1[idx], self.gamma2[idx],
                  self.labels [idx])
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
        self.activation5 = nn.GELU()

    def forward(self, x):
        out = self.activation1(self.il(x))
        out = self.activation2(self.hl1(out))
        out = self.activation3(self.hl2(out))
        out = self.activation4(self.hl3(out))
        out = self.activation5(self.hl4(out))
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
            nn.Linear(16, output_size)
        )

    def forward(self, input_psd, input_anat):
        # Propagación a través de las capas correspondientes a InputPSD
        out0 = self.NN0(input_psd)

        # Propagación a través de las capas correspondientes a InputAnat
        out1 = self.NN1(input_anat)

        # Concatenar las salidas
        concatenated_output = torch.cat((out0, out1), dim=1)

        # Propagación a través de la capa de salida
        final_output = self.output_layer(concatenated_output)

        return final_output


#%% To reset wights
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


#%% Create Dataset for graph regression
from torch_geometric.data import Data, Batch
import networkx as nx
def Dataset_graph(features, labels, connectome, task):
    features = np.array(features,dtype=float)
    data_list=[]
    le = LabelEncoder()
    encoded=le.fit_transform(labels)
    task=torch.FloatTensor(task[:,np.newaxis,np.newaxis])
    for i in range(len(features)):
        x=torch.FloatTensor(features[i,:,:].T)
        edge_index,edge_wight=from_scipy_sparse_matrix(sparse.csr_matrix(connectome[i,:,:]))
        edge_index=torch.from_numpy(edge_index.numpy())
        # print(edge_index.shape)
        edge_wight=torch.from_numpy(edge_wight.numpy())
        data=Data(x=x,edge_index=edge_index,edge_attr=edge_wight,y=task[i])
        data_list.append(data)
    # print(data)
    dataset=Batch.from_data_list(data_list)
    return dataset

#%% Graph neural net w/o pull
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
class GCN(torch.nn.Module):
    def __init__(self, num_node_features,  hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)  
        self.il = nn.Linear(hidden_channels, 64)
        self.hl1 = nn.Linear(64, 16)
        self.hl2 = nn.Linear(16, 16)
        self.ol = nn.Linear(16, 1)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.ReLU()
        # self.pooling=nn.AdaptiveAvgPool2d((hidden_channels,2))
        # self.activation3 = nn.ELU()
        # self.activation4 = nn.SELU()
        # self.activation5 = nn.GELU()

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = self.activation1(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # x= self.pooling(x)
        # print(x.shape)
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        out = self.activation1(self.il(x))
        out = self.activation2(self.hl1(out))
        out = self.ol(out)
        
        return out




