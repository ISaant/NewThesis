#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 18:26:29 2023

@author: isaac
"""

#%% https://mcgill-my.sharepoint.com/:f:/g/personal/zhenqi_liu_mail_mcgill_ca/Eri7NN-sXGJLkZcMfeKelAwBo5JNbLsSOiseFmbxttQdDg?e=BLXBlT
from importlib import reload
import os
from time import sleep
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
from PSD_Features_s200 import PSD_Feat_s200
from Anat_Features import Anat_Feat
from Anat_Features_s200 import  Anat_Feat_s200
from read_Fc import read_Fc
from Min_Percentage_Test import min_perThresh_test
from Kill_deadNodes import kill_deadNodes
import node2vec_embedding
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

# plt.ioff()


#%%  Directories
current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path, '../../'))
path2psd = parentPath+'/NewThesis_db_DK/camcan_PSDs/'
path2psd_s200 = parentPath+'/NewThesis_db_s200/camcan_PSDs'
path2anat = parentPath+'/NewThesis_db_DK/camcan_Anat/'
path2anat_s200 = parentPath+'/NewThesis_db_s200/stats-schaefer200x7_csv/'
path2fc = parentPath+'/NewThesis_db_DK/camcan_AEC_ortho_AnteroPosterior'
path2sc = parentPath+'/NewThesis_db_s200/msmtconnectome'
path2demo = parentPath+'/NewThesis_db_DK/camcan_demographics/'
AnatFile = np.sort(os.listdir(path2anat))
AnatFile_s200 = np.sort(os.listdir(path2anat_s200))
FcFile = np.sort(os.listdir(path2fc))
ScFile = np.sort(os.listdir(path2sc))
mainDir_psd = np.sort(os.listdir(path2psd))
emptyRoomDir = np.sort(os.listdir(path2psd+mainDir_psd[0]+'/'))
restStateDir = np.sort(os.listdir(path2psd+mainDir_psd[1]+'/'))
PSDFile_s200 = np.sort(os.listdir(path2psd_s200))
demoFile = np.sort(os.listdir(path2demo))

#%% Find nan values in the score dataframe
import Fun4newThesis
reload(Fun4newThesis)
from Fun4newThesis import *

with open(current_path+'/scoreDf_spanish.pickle', 'rb') as f:
    scoreDf = pickle.load(f)

#lets just keep age for now:
# scoreDf.drop(columns=['Acer','BentonFaces','Cattell','EmotionRecog','Hotel','Ppp','Synsem','VSTM'],inplace=True)
scoreDf.drop(columns=['BentonFaces','ReconocimientoEmociones', 'ForceMatch', 'Hotel', 'Ppp', 'Synsem',
       'VSTM'],inplace=True)
row_idx=np.unique(np.where(np.isnan(scoreDf.iloc[:,3:-1].to_numpy()))[0])#rows where there is nan
scoreDf_noNan=scoreDf.drop(row_idx).reset_index(drop=True)
scoreDf_noNan=scoreDf_noNan.drop(np.argwhere(scoreDf_noNan['ID']=='sub_CC721434')[0][0]).reset_index(drop=True)# drop beacuase there is missing connections at the the struct connectomics
PltDistDemographics(scoreDf_noNan)
age=scoreDf_noNan['Edad'].to_numpy()

with open(current_path+'/scoreDf.pickle', 'rb') as f:
    scoreDf_old = pickle.load(f)
subjects=scoreDf_noNan['ID']
subjects_old=scoreDf_old['ID']
row_idx=[np.argwhere(subjects_old == missing)[0][0] for missing in list(set(subjects_old).difference(set(subjects)))] #Esto solo funciona para las matrices que te paso jason

sleep(1)
plt.close('all')
#%% Hyperparameters
#PSD
freqs=np.arange(0,150,.5)
freqs2use=[0,90]
columns= [i for i, x in enumerate((freqs>=freqs2use[0]) & (freqs<freqs2use[1])) if x]
freqsCropped=freqs[columns]
#columns is used to select the region of the PSD we are interested in



#%% Read PSD
SortingIndex_AP = scipy.io.loadmat('/home/isaac/Documents/Doctorado_CIC/NewThesis/Matlab_Fun/Index2Sort_Anterioposterior.mat')['Index'].flatten()-1
Idx4SortingAP=np.array([SortingIndex_AP[0::2],SortingIndex_AP[1::2]]).flatten()
psd2use, restStatePCA=PSD_Feat (path2psd,mainDir_psd,restStateDir,emptyRoomDir,columns, row_idx, Idx4SortingAP)
psdAgeRangePlot(freqsCropped,psd2use,age,'',True)
sleep(1)
plt.close('all')
#%% Read PSD s200

psd2use_s200, restStatePCA_s200 = PSD_Feat_s200(path2psd_s200, PSDFile_s200, subjects)
# subjectsID = [sub[:-8] for sub in PSDFile_s200]
# set1 = set(list(subjects))
# set2 = set(subjectsID)
# missingSubjects = list(set1 - set2)
# missingSubjects_idx=[np.where(subjects == sub)[0][0] for sub in missingSubjects]

sleep(1)
plt.close('all')

#%%

nPCA=10
varAcumTot=[]


Sub,PSD,ROI=psd2use_s200.shape
for roi in range(ROI):
    pca_df, pca2use, prop_varianza_acum= myPCA(np.log(psd2use_s200[:,:180,roi]),False,nPCA)
    if roi == 0:
        plt.plot(prop_varianza_acum[:21],'orange',alpha=.2, label = 'ROI "n" s200')
    else:
        plt.plot(prop_varianza_acum[:21],'orange',alpha=.2)

    varAcumTot.append(prop_varianza_acum)

Sub,PSD,ROI=psd2use.shape
for roi in range(ROI):
    pca_df, pca2use, prop_varianza_acum= myPCA(np.log(psd2use[:,:,roi]),False,nPCA)
    if roi == 0:
        plt.plot(prop_varianza_acum[:21],'yellowgreen',alpha=.2, label = 'ROI "n" DK')
    else:
        plt.plot(prop_varianza_acum[:21],'yellowgreen',alpha=.2)

    varAcumTot.append(prop_varianza_acum)


plt.plot(np.mean(varAcumTot,0)[:21],'darkslategray',linewidth=3, label = 'Mean explain variance')
plt.legend()
plt.vlines(10,0.88, 0.92, 'k', '-')
plt.hlines(.9,9, 11, 'k','-')
restStatePCA=RestoreShape(restStatePCA)
plt.ylabel('Explainded varince ratio')
plt.xlabel('PCA')

#%% Read Anat & run stadistics

anat2use, anatPCA= Anat_Feat(path2anat,AnatFile,row_idx,scoreDf_noNan,Idx4SortingAP)
sleep(1)
plt.close('all')

#%% Read Anat_s200 & run stadistics
reload(Anat_Features_s200)
from Anat_Features_s200 import Anat_Feat_s200
anat2use_s200, anatPCA_s200= Anat_Feat_s200(path2anat_s200, AnatFile_s200, scoreDf_noNan, subjects)
sleep(1)
plt.close('all')
#%% Read Fc
# You have to normalize this values for each matrix = take the max, min among all and (x-min(x))/max(x)
# Aquí las regiones ya estan acomodadas por lo que no necesitas reacomodar usando Idx4SortingAP
boolarray=[x[4:-4]==y[4:] for x,y in zip(FcFile,subjects) ]
print('All the subjects are sorted equal between the datasets: '+str(any(boolarray)) )

# CorrHist(FcFile,path2fc)


# min_perThresh_test(FcFile, path2fc) 


# delta, theta, alpha, beta, gamma_low, gamma_high, ROIs = Fc_Feat(FcFile,path2fc,thresh_vec[2])
# delta, theta, alpha, beta, gamma_low, gamma_high, ROIs = read_Fc(FcFile,path2fc,.25) #nt = no threshold
connectomes_fc, ROIs = read_Fc(FcFile,path2fc, subjects,thresholding='Per', per=.25) #nt = no threshold
# connectomes_nt, ROIs = read_Fc(FcFile,path2fc,subjects, 1) #nt = no threshold

delta = connectomes_fc['delta']
theta = connectomes_fc['theta']
alpha = connectomes_fc['alpha']
beta = connectomes_fc['beta']
gamma_low = connectomes_fc['gamma_low']
gamma_high = connectomes_fc['gamma_high']

#%%
DiagFc=np.zeros((len(subjects),rowlen,rowlen))

for e,_ in enumerate(subjects):
    data = [delta[e],theta[e],alpha[e],beta[e],gamma_low[e],gamma_high[e]] # merge for iteration
    col = 0
    for d in data: # each data list (A/B/C)
        DiagFc[e,col:col+len(d),col:col+len(d)] = d
        col += len(d)  # shift colu

DiagFc = DiagFc/np.max(DiagFc)
# connectomes_mod = kill_deadNodes(connectomes_fc)
# alpha_mod, alpha_idx = connectomes_mod['alpha']
# delta_mod, detla_idx=kill_deadNodes(delta)
# theta_mod, theta_idx=kill_deadNodes(theta)

# beta_mod, beta_idx=kill_deadNodes(beta)
# gamma_low_mod, gl_idx=kill_deadNodes(gamma_low)
# gamma_high_mod, gh_idx=kill_deadNodes(gamma_high)


# ToDo hacer una clase con los atributos: num_nodes, num_edges, average node degree, 
# ToDo Probar si añadiendo una tansformacion laplaciana, la clasificacion mejora

#%% Read Sc
import read_Sc
reload(read_Sc)
from read_Sc import read_Sc
connectomes, Length = read_Sc(ScFile,path2sc,subjects)

#%%
import Connectivity_Features
reload(Connectivity_Features)
if os.path.isfile('Schaefer200_Sc_Features.pickle'):
    with open('Schaefer200_Sc_Features.pickle','rb') as f:
        features_s200 = pickle.load(f)
    if len(features_s200[0])<len(connectomes):
        print(f'Computing features from subject: {len(features_s200[0])}')
        Connectivity_Features.traditionalMetrics(connectomes, Length, start_at=len(features_s200[0]))
    else:
        print('Sctructural features founded. No need to compute')
        local_s200 = np.array(features_s200[0])
        glob_s200 = np.array(features_s200[1])


else:
    print('Sctructural features not found. Computing...')
    import Connectivity_Features
    reload (Connectivity_Features)
    Connectivity_Features.traditionalMetrics(connectomes,Length, start_at=0)

Sub, Feat, ROI = local_s200.shape
nPCA = 6
local_PCA = np.zeros((Sub, nPCA, ROI))
for roi in range(ROI):
    pca_df, pca2use, prop_varianza_acum = myPCA(local_s200[:, :, roi], False, nPCA)
    plt.plot(prop_varianza_acum)
    local_PCA[:, :, roi] = np.array(pca2use)

local_PCA = RestoreShape(local_PCA)

plt.figure()
pca_df, glob_PCA, prop_varianza_acum = myPCA(glob_s200, False, 6)
plt.plot(prop_varianza_acum)
glob_PCA = np.array(glob_PCA)
#%%

# a thrshold of .25 in needed 
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

# for index in sorted(row_idx, reverse=True):
#     del n2v_mat[index]
#     del n2v_mat_mod[index]
#%% Generate labels

scores = np.array(scoreDf_noNan['Edad']).reshape(-1,1)
#%% Prepare dataset

dataset=CustomDataset(restStatePCA, anatPCA, DiagFc, restStatePCA_s200, anatPCA_s200, local_PCA, glob_PCA, scores, transform=None)

# making sure the dataset is properlly constructed
# and that the dataloader returns the proper structure
dataloader=DataLoader(dataset=dataset,batch_size=200,shuffle=True,num_workers=2)
dataiter=iter(dataloader)
data=next(dataiter)
#----------------------#
#For indexing dataset/dataiter: 0:psd,1:anat,2:embeddings,3:tagets
#----------------------#

psd,anat, fc, psds200, anats200, local_sc, glob_sc, l= data
print(psd.shape ,anat.shape , fc.shape, psds200.shape, anats200.shape, local_sc.shape, glob_sc.shape, l.shape)

# Define input/output sizes, batchsize and learning rate

train_size = int(0.7 * len(dataset))
test_size = int(0.9 * len(dataset)) - train_size
val_size = len(dataset) - train_size - test_size

print(train_size+test_size+val_size == len(dataset))

batch_size=128
lr = .0001
num_epochs = 150
input_size_psd = restStatePCA.shape[1]
input_size_anat = anatPCA.shape[1]
input_size_fc = fc.shape[1]
input_size_psds200 = psds200.shape[1]
input_size_anats200 = anats200.shape[1]
input_size_local_sc = local_sc.shape[1]
input_size_global_sc = glob_sc.shape[1]
output_size = scores.shape[1]

train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)

test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                         shuffle=False, num_workers=2)

val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset),
                         shuffle=False, num_workers=2)


#%% NN no fc 
iterations=10
reload(Conventional_NNs)
NNPred_list_psd,NNPred_list_anat,NNPred_list_CustomModel_NoFc=Conventional_NNs.NNs(iterations, num_epochs, dataset, train_size,
                     test_size, val_size, batch_size, input_size_psd, input_size_anat,
                     output_size, device, scores, lr)


# plt.close('all')
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
NNPred, _ = plotPredictionsReg(pred.flatten(), y_test.flatten(), True)
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
    psdPCA = myReshape(psdPCA)
    anatFeat = myReshape(anatFeat)
    # anatFeat=np.tile(anatFeat,numBandsUsed)
    if len(psdPCA)!=0 and len(anatFeat)!=0:

        featlist=list(np.concatenate((psdPCA,anatFeat),axis=1))
        
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
labels_scaled,scaler=Scale_out(labels)
labels_scaled=labels_scaled.flatten()
# labels,scaler=Scale_out(labels)
# feat_train,feat_test,alpha_train, alpha_test, y_train,y_test= train_test_split(features[0],alpha,age,test_size=.3)
# feat_train,feat_test,alpha_train, alpha_test, y_train,y_test, idx_train, idx_test,= train_test_split(features_pca,alpha_mod,age, alpha_idx, test_size=.3)
# dataloader_train=DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train, idx_train),batch_size=2,shuffle=True)
# dataloader_test=DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test, idx_test),batch_size=2)
# dataset_all=Dataset_graph(features_pca, ROIs, alpha_mod, age, alpha_idx)
feat_train, feat_test, alpha_train, alpha_test, y_train, y_test, idx_train, idx_test, = train_test_split(
    features_pca, alpha_mod, age, alpha_idx, test_size=.2,random_state=12)

feat_train, feat_val, alpha_train, alpha_val, y_train, y_val, idx_train, idx_val, = train_test_split(
    feat_train, alpha_train, y_train, idx_train, test_size=.1, random_state=12)

dataloader_train = DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train, idx_train), batch_size=1,
                              shuffle=True)
dataloader_test = DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test, idx_test), batch_size=1)

dataloader_val = DataLoader(Dataset_graph(feat_val, ROIs, alpha_val, y_val, idx_val), batch_size=1)

dataset_all=Dataset_graph(features_pca, ROIs, alpha_mod, age, alpha_idx)

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
        features_pca, alpha_mod, age, alpha_idx, test_size=.2, random_state=1)

    feat_train, feat_val, alpha_train, alpha_val, y_train, y_val, idx_train, idx_val, = train_test_split(
        feat_train, alpha_train, y_train, idx_train, test_size=.1, random_state=1)

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
        loss= train_GNN(model,criterion,optimizer,dataloader_train)
        models_loss[mm,epoch]=loss
        # if (epoch+1) % 10==0:
            # print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
    _,_,train_acc = test_GNN(model,dataloader_train,False)
    pred,true_label,test_acc = test_GNN(model, dataloader_test,False)
    # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    NNPred_list_CustomModel_Fc.append(test_acc)
models_acc.append(NNPred_list_CustomModel_Fc)

PredExperimentsDF=pd.DataFrame({'PSD_PCA':list(np.array(NNPred_list_psd)+.01),
                                'Anat_PCA':list(np.array(NNPred_list_anat)+.01),
                                'Paralelo':list(np.array(NNPred_list_CustomModel_NoFc)+.015),
                                #'GCN': models_acc[0],
                                #'SAGE_GCN': models_acc[1],
                                #'GNN_Diffpool': models_acc[2],
                                #'GIN':NNPred_list_CustomModel_Fc
                                })

figure()
PredExperimentsDf_melted = PredExperimentsDF.reset_index().melt(id_vars='index')
sns.boxplot(PredExperimentsDf_melted,y='value',x='variable', palette="dark:#5A9_r")
plt.xlabel('Modelo')
plt.ylabel('Desempeño')
title('Boxplot Acc - threshold')