#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 18:26:29 2023

@author: isaac
"""

#%%
import os
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
# os.chdir('/export03/data/Santiago/NewThesis/Python_Fun')
import pickle
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import scipy
import torch.optim as optim
import Conventional_NNs
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from tqdm import tqdm
from matplotlib.pyplot import plot, figure, title
from Fun4newThesis import *
from FunClassifiers4newThesis_pytorch import *
from sklearn.linear_model import Lasso
from copy import copy 
from sklearn.model_selection import train_test_split
from PSD_Features import PSD_Feat
from Anat_Features import Anat_Feat
from read_Fc import read_Fc
from Min_Percentage_Test import min_perThresh_test
from Kill_deadNodes import kill_deadNodes
import node2vec_embedding
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
#%% Hyperparameters

#PSD
freqs=np.arange(0,150,.5)
freqs2use=[0,90]
columns= [i for i, x in enumerate((freqs>=freqs2use[0]) & (freqs<freqs2use[1])) if x]
freqsCropped=freqs[columns]
#columns is used to select the region of the PSD we are interested in


#%%  Directories
current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path,'../../'))
path2psd = parentPath+'/NewThesis_db/camcan_PSDs/'
path2anat =  parentPath+'/NewThesis_db/camcan_Anat/'
path2fc= parentPath+'/NewThesis_db/camcan_AEC_ortho_AnteroPosterior'
path2demo=parentPath+'/NewThesis_db/camcan_demographics/'
AnatFile=np.sort(os.listdir(path2anat))
FcFile=np.sort(os.listdir(path2fc))
mainDir_psd=np.sort(os.listdir(path2psd))
emptyRoomDir=np.sort(os.listdir(path2psd+mainDir_psd[0]+'/'))
restStateDir=np.sort(os.listdir(path2psd+mainDir_psd[1]+'/'))
demoFile=np.sort(os.listdir(path2demo))


#%% Find nan values in the score dataframe
with open(current_path+'/scoreDf.pickle', 'rb') as f:
    scoreDf = pickle.load(f)

#lets just keep age for now:
scoreDf.drop(columns=['Acer','BentonFaces','Cattell','EmotionRecog','Hotel','Ppp','Synsem','VSTM'],inplace=True)
row_idx=np.where(np.isnan(scoreDf.iloc[:,3:-1].to_numpy()))[0] #rows where there is nan

scoreDf_noNan=scoreDf.drop(row_idx)
age=scoreDf['Age'].to_numpy()
#%% Demographics
demographics=pd.read_csv(path2demo+demoFile[0])
subjects=demographics['CCID']



#%% Read PSD

psd2use, restStatePCA=PSD_Feat (path2psd,mainDir_psd,restStateDir,emptyRoomDir,columns, row_idx)


#%% Read Anat & run stadistics

anat2use,Anat_aranged, anatPCA= Anat_Feat(path2anat,AnatFile,row_idx,scoreDf_noNan)

#%% Read Fc
# You have to normalize this values for each matrix

boolarray=[x[4:-4]==y[4:] for x,y in zip(FcFile,subjects) ]
print('All the subjects are sorted equal between the datasets: '+str(any(boolarray)) )

# CorrHist(FcFile,path2fc)


# min_perThresh_test(FcFile, path2fc) 


# delta, theta, alpha, beta, gamma_low, gamma_high, ROIs = Fc_Feat(FcFile,path2fc,thresh_vec[2])
delta, theta, alpha, beta, gamma_low, gamma_high, ROIs = read_Fc(FcFile,path2fc,.25) #nt = no threshold
delta_mod,detla_idx=kill_deadNodes(delta)
theta_mod,theta_idx=kill_deadNodes(theta)
alpha_mod,alpha_idx=kill_deadNodes(alpha)
beta_mod,beta_idx=kill_deadNodes(beta)
gamma_low_mod,gl_idx=kill_deadNodes(gamma_low)
gamma_high_mod,gh_idx=kill_deadNodes(gamma_high)


# ToDo hacer una clase con los atributos: num_nodes, num_edges, average node degree, 
# ToDo Probar si añadiendo una tansformacion laplaciana, la clasificacion mejora

#%%
emb_dim=128

# for file in listoffiles:
if os.path.isfile('embeddings.pickle'):
    print('No need to compute embeddings')
    with open('embeddings.pickle','rb') as f:
        emb_loss = pickle.load(f)
    n2v_mat = emb_loss[0]
    loss_mat = emb_loss[1]

else:
    print('Embeddings was not found, starting computation')
    n2v_mat, loss_mat = node2vec_embedding.n2v_embedding(alpha_mod, device=device, q=1.5, embedding_dim=emb_dim)
    emb_loss = [n2v_mat, loss_mat]
    with open('embeddings.pickle', 'wb') as f:
        pickle.dump(emb_loss, f)

# add cero array to the disjointed nodes
n2v_mat_mod = []
for emb, idx in zip(n2v_mat,alpha_idx):
    for i in idx:
        emb = np.insert(arr=emb, obj=i, values=0, axis=0)
    # print(emb.shape)
    n2v_mat_mod.append(emb)
#%% Generate labels

# scores=scoreDf_noNan[scoreDf.columns[[1,3,4,5,6,7,8,9,10]]].to_numpy()
scores=scoreDf_noNan[scoreDf.columns[[1]]].to_numpy()
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler.fit(scores)
# labels=scaler.transform(scores)
labels= scores

#%% Prepare dataset

dataset=CustomDataset(restStatePCA, anatPCA, n2v_mat, n2v_mat_mod, labels, transform=None)

# making sure the dataset is properlly constructed
# and that the dataloader returns the proper structure
dataloader=DataLoader(dataset=dataset,batch_size=200,shuffle=True,num_workers=2)
dataiter=iter(dataloader)
data=next(dataiter)
#----------------------#
#For indexing dataset/dataiter: 0:psd,1:anat,2:embeddings,3:tagets
#----------------------#

psd, anat, emb, emb_z, l= data
print(psd.shape, anat.shape, emb.shape, emb_z.shape, l.shape)

# Define input/output sizes, batchsize and learning rate

train_size = int(0.6 * len(dataset))
test_size = int(0.9 * len(dataset)) - train_size
val_size = len(dataset) - train_size - test_size

print(train_size+test_size+val_size == len(dataset))

batch_size=128
lr = .0001
num_epochs = 150
input_size_psd = restStatePCA.shape[1]
input_size_anat = anatPCA.shape[1]
input_size_emb = emb.shape[1]
input_size_emb_z = emb_z.shape[1]
output_size = labels.shape[1]

train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)

test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                         shuffle=False, num_workers=2)

val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset),
                         shuffle=False, num_workers=2)


#%% NN no fc 
iterations=1
NNPred_list_psd,NNPred_list_anat,NNPred_list_CustomModel_NoFc=Conventional_NNs.NNs(iterations, num_epochs, dataset, train_size,
                     test_size, val_size, batch_size, input_size_psd, input_size_anat,
                     output_size, device, lr)

plt.close('all')
#%% Test NN for graph embeddings

model = NeuralNet4Graph_emb(input_size_emb, output_size).to(device)
print(model._get_name())
# Puedes imprimir el resumen del modelo si lo deseas
# print(model)
if 'model' in globals():
    model.apply(weight_reset)
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
var_pos = 2
train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
mse_test, pred = test_ANN(model,test_loader, var_pos)
pred=np.array(pred)
y_test = test_dataset[:][-1].numpy()
NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), True)
print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')



#%% Dataset Graph
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.loader import DataLoader
def select_feat_noPCA(psdFeat,bandOfInt,anatFeat):
    if bandOfInt.ndim ==2:
        numBandsUsed,_=bandOfInt.shape
    else:
        bandOfInt=bandOfInt[np.newaxis,]
        numBandsUsed=1
    # anatFeat=np.tile(anatFeat,numBandsUsed)
    if len(psdFeat)!=0 and len(anatFeat)!=0:
        featlist=[]
        for i in range(numBandsUsed):
            featlist.append(list(np.concatenate((np.log(psdFeat[:,(freqsCropped>=bandOfInt[i,0])&(freqsCropped<=bandOfInt[i,1]),:]),myReshape(anatFeat)),axis=1)))
        
        return featlist
    elif len(psdFeat)!=0 and len(anatFeat)==0:
        return np.log(psdFeat[:,(freqsCropped>=bandOfInt[0])&(freqsCropped<=bandOfInt[1]),:])
    else:
        return myReshape(anatFeat)

def select_feat_psdPca(psdPCA,anatFeat):
    psdPCA=myReshape(psdPCA)
    # anatFeat=np.tile(anatFeat,numBandsUsed)
    if len(psdPCA)!=0 and len(anatFeat)!=0:

       
        featlist=list(np.concatenate((psdPCA,myReshape(anatFeat)),axis=1))
        
        return featlist
    elif len(psdPCA)!=0 and len(anatFeat)==0:
        # return psdPCA[:,(freqsCropped>=bandOfInt[0])&(freqsCropped<=bandOfInt[1]),:]
        return list(psdPCA)
    else:
        return list(myReshape(anatFeat))


def select_feat_emb(emb, anatFeat):
    if len(emb) != 0 and len(anatFeat) != 0:
        emb = np.swapaxes(np.array(emb), 1, 2)
        featlist = list(np.concatenate((emb, myReshape(anatFeat)), axis=1))
        return featlist
    elif len(emb) != 0 and len(anatFeat) == 0:
        emb = np.swapaxes(np.array(emb), 1, 2)
        return list(emb)
    else:
        return list(myReshape(anatFeat))

# features= select_feat_noPCA(psd2use,np.array([8,12]),anatPCA)
features_pca= select_feat_psdPca(restStatePCA, anatPCA)
features_emb= select_feat_emb(n2v_mat_mod, [])
# feat_train,feat_test,alpha_train, alpha_test, y_train,y_test= train_test_split(features[0],alpha,age,test_size=.3)
# feat_train,feat_test,alpha_train, alpha_test, y_train,y_test, idx_train, idx_test,= train_test_split(features_pca,alpha_mod,age, alpha_idx, test_size=.3)
# dataloader_train=DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train, idx_train),batch_size=2,shuffle=True)
# dataloader_test=DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test, idx_test),batch_size=2)
# dataset_all=Dataset_graph(features_pca, ROIs, alpha_mod, age, alpha_idx)
feat_train, feat_test, alpha_train, alpha_test, y_train, y_test, idx_train, idx_test, = train_test_split(
    features_emb, alpha_mod, age, alpha_idx, test_size=.3,random_state=12)

feat_train, feat_val, alpha_train, alpha_val, y_train, y_val, idx_train, idx_val, = train_test_split(
    feat_train, alpha_train, y_train, idx_train, test_size=.1, random_state=12)

dataloader_train = DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train, idx_train), batch_size=1,
                              shuffle=True)
dataloader_test = DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test, idx_test), batch_size=1)

dataloader_val = DataLoader(Dataset_graph(feat_val, ROIs, alpha_val, y_val, idx_val), batch_size=1)

dataset_all=Dataset_graph(features_emb, ROIs, alpha_mod, age, alpha_idx)

for i in range(10):
    data = dataset_all[i]
    print(data, alpha_idx[i].shape)

# vis = to_networkx(data)
# node_labels = data.y.numpy()
# plt.figure(1,figsize=(15,13))
# nx.draw(vis, cmap=plt.get_cmap('Set3'),node_size=70, linewidths=6)


#%%
# dataloader_rw=DataLoader(dataset=dataset_rw,batch_size=3,shuffle=True,num_workers=2)
# dataiter=iter(dataloader_rw)
# data=next(dataiter)
# added, concat, l= data
# print(psd.shape, anat.shape,l.shape)
#%%

#### 

#Ninguno de tus modelos usa edge_attr. Implementalo mas adelante

####



model1 = GCN(data.x.shape[1],hidden_channels=12, lin=True)
model2 = SAGE_GCN (data.x.shape[1],hidden_channels=32, lin=True)
model3 = GNN_DiffPool(data.x.shape[1])

models = [model1,model2,model3]
models_acc=[]
num_epochs= 200
models_loss= np.zeros((3,num_epochs))

for i in tqdm(range(iterations)):

    feat_train, feat_test, alpha_train, alpha_test, y_train, y_test, idx_train, idx_test, = train_test_split(
        features_emb, alpha_mod, age, alpha_idx, test_size=.3, random_state=12)

    feat_train, feat_val, alpha_train, alpha_val, y_train, y_val, idx_train, idx_val, = train_test_split(
        feat_train, alpha_train, y_train, idx_train, test_size=.1, random_state=12)

    dataloader_train = DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train, idx_train), batch_size=1,
                                  shuffle=True)
    dataloader_test = DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test, idx_test), batch_size=1)

    dataloader_val = DataLoader(Dataset_graph(feat_val, ROIs, alpha_val, y_val, idx_val), batch_size=1)

    # Derive ratio of correct predictions.

    NNPred_list_CustomModel_Fc=[]
    for mm, model in enumerate(models):
        print(model._get_name())
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.MSELoss()
        if 'model' in globals():
            model.apply(weight_reset)
        train_GNN(model, criterion, optimizer, dataloader_train, dataloader_val, num_epochs)
        mse_test, pred, test_acc = test_GNN(model, dataloader_test, True)
        print(f'Test_Acc: {test_acc:4f}, Test_MSE: {mse_test}')
        NNPred_list_CustomModel_Fc.append(test_acc)
    models_acc.append(NNPred_list_CustomModel_Fc)




#%%

model = GNN_GIN(data.x.shape[1], train_eps=False, output_size=32)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
criterion = nn.MSELoss()    
NNPred_list_CustomModel_Fc=[]
for i in tqdm(range (iterations)):
    feat_train,feat_test,alpha_train, alpha_test, y_train,y_test= train_test_split(features_pca,alpha,age,test_size=.3)
    dataloader_train=DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train),batch_size=10,shuffle=True,num_workers=2)
    dataloader_test=DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test),batch_size=10,num_workers=2)


    if 'model' in globals():
        # print('yes')
        model.apply(weight_reset)
    n_total_steps = len(dataloader_train)
    for epoch in range(1, num_epochs):
        loss=train(model,criterion,optimizer,dataloader_train)
        models_loss[mm,epoch]=loss
        # if (epoch+1) % 10==0:
            # print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
    _,_,train_acc = test(model,dataloader_train,False)
    pred,true_label,test_acc = test(model, dataloader_test,False)
    # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    NNPred_list_CustomModel_Fc.append(test_acc)
models_acc.append(NNPred_list_CustomModel_Fc)

PredExperimentsDF=pd.DataFrame({'PSD_PCA':NNPred_list_psd,
                                'Anat_PCA':NNPred_list_anat,
                                'Parallel':NNPred_list_CustomModel_NoFc,
                                'GCN': models_acc[0],
                                'SAGE_GCN': models_acc[1],
                                'GNN_Diffpool': models_acc[2],
                                'GIN':NNPred_list_CustomModel_Fc })

figure()
PredExperimentsDf_melted = PredExperimentsDF.reset_index().melt(id_vars='index')
sns.boxplot(PredExperimentsDf_melted,y='value',x='variable')

title('Boxplot Acc - threshold')