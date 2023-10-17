#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 18:26:29 2023

@author: isaac
"""

import os
# os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
import pickle
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import scipy
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from matplotlib.pyplot import plot, figure, title
from Fun4newThesis import *
from FunClassifiers4newThesis_pytorch import *
from sklearn.linear_model import Lasso
from copy import copy 
from sklearn.model_selection import train_test_split
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
#%% Demographics
demographics=pd.read_csv(path2demo+demoFile[0])
subjects=demographics['CCID']

#%% Read PSD

for e,file in enumerate(tqdm(restStateDir)):
    matrix=myReshape(pd.read_csv(path2psd+mainDir_psd[1]+'/'+file,header=None).to_numpy())[np.newaxis, :]
    if e == 0:
        restStateAll=matrix 
        continue
    restStateAll=np.concatenate((restStateAll,matrix))
emptyRoom=pd.read_csv(path2psd+mainDir_psd[0]+'/'+emptyRoomDir[0],header=None).to_numpy()
emptyRoom = myReshape(emptyRoom) #reshape into [Subjects,PSD,ROI]
emptyRoomCropped = emptyRoom[:,columns,:]
restState=np.mean(restStateAll,axis=0)
psd2use=np.delete(restState,row_idx,axis=0)[:,columns,:] #remove index with nan and select the band-width of interest

# PCA on PSD
Sub,PSD,ROI=psd2use.shape
nPCA=10
restStatePCA=np.zeros((Sub,nPCA,ROI))
for roi in range(ROI):
    pca_df, pca2use, prop_varianza_acum= myPCA(np.log(psd2use[:,:,roi]),False,nPCA)
    plt.plot(prop_varianza_acum[:10])
    restStatePCA[:,:,roi]=np.array(pca2use)
    
restStatePCA=RestoreShape(restStatePCA)

#%% Read Anat & run stadistics

AnatomicalFeatures=pd.read_csv(path2anat+AnatFile[0])
AnatomicalFeatures.drop(columns=['Unnamed: 0'],inplace=True)
anat2use=myReshape(np.delete(AnatomicalFeatures, row_idx,axis=0))

Sub,PSD,ROI=anat2use.shape
nPCA=5
anatPCA=np.zeros((Sub,nPCA,ROI))
for roi in range(ROI):
    pca_df, pca2use, prop_varianza_acum= myPCA(anat2use[:,:,roi],False,nPCA)
    plt.plot(prop_varianza_acum[:10])
    anatPCA[:,:,roi]=np.array(pca2use)

anat2use=RestoreShape(anat2use)
anatPCA=RestoreShape(anatPCA)


age=scoreDf_noNan['Age'].to_numpy()
Reg=[]
for i in tqdm(range(200)):
    x_train, x_test, y_train,y_test,_,_=Split(anat2use,age,.3,seed=i)
    x_train=Scale(x_train)
    x_test=Scale(x_test)
    model = Lasso(alpha=.2)
    model.fit(x_train, y_train)
    pred_Lasso=model.predict(x_test)
    LassoPred=plotPredictionsReg(pred_Lasso,y_test,False)
    Reg.append(LassoPred)
    x_test=myReshape(x_test)
    sub,ft,ROI=x_test.shape
    matPred=np.zeros((ft,ft))
    matPred[:]=np.nan

    for j in np.arange(0,ft-1):
        for k in np.arange(j+1,ft):
            # print(j,k)
            cp=copy(x_test)
            cp[:,[j,k],:]=cp[:,[k,j],:]
            cp=RestoreShape(cp)
            cp_pred_Lasso=model.predict(cp)
            cp_LassoPred=plotPredictionsReg(cp_pred_Lasso,y_test,False)
            matPred[j,k]=cp_LassoPred
            matPred[k,j]=cp_LassoPred
            
    matPred=np.nanmean((LassoPred-matPred),axis=0)[np.newaxis,]
    if i == 0:
        MatPred=matPred
        continue
    MatPred=np.concatenate((MatPred,matPred))

MatPredDf=pd.DataFrame(MatPred,columns=["NumVert" , "SurfArea" ,
                                            "GrayVol" , "ThickAvg",
                                            "ThickStd", "MeanCurv",
                                            "GausCurv", "FoldInd" ,
                                            "CurvInd" ])

MatPredDf_melted = MatPredDf.reset_index().melt(id_vars='index')
figure()
sns.kdeplot(MatPredDf_melted,x='value',hue='variable',fill=True, 
            common_norm=False, palette="rainbow",alpha=.5, linewidth=1)

title('Feature importance, "Flip" approach')

anat2use=RestoreShape(np.delete(myReshape(anat2use),[0,1,7],axis=1))
# Rearange for NN
cont=0
Anat_aranged=np.zeros((anat2use.shape)) #!!! YA NO PUEDES RESTAURAR A (SUB,ANAT,ROI) USANDO RESTORESHAPE
for i in range (6):
    for j in np.arange(0,6*68,6):
        # print(i+j)
        Anat_aranged[:,cont]=anat2use[:,i+j]
        cont+=1
del model
#%% Read Fc
# You have to normalize this values for each matrix

boolarray=[x[4:-4]==y[4:] for x,y in zip(FcFile,subjects) ]

print('All the subjects are sorted equal between the datasets: '+str(any(boolarray)) )

'''Aqui debes decidir si usar k vecinos o umbralizar. knn nos dara una matriz sparce uniforme'''
for e,file in enumerate(FcFile):
    mat = scipy.io.loadmat(path2fc+'/'+file)
    fcMatrix=np.arctanh(knn_graph(mat['TF_Expand_Matrix_Sorted'],Nneighbours=8))
    # fcMatrix=np.arctanh(threshold(mat['TF_Expand_Matrix_Sorted'],tresh=.3))

    if e==0:
        delta=fcMatrix[:,:,0][np.newaxis,:,:]
        theta=fcMatrix[:,:,1][np.newaxis,:,:]
        alpha=fcMatrix[:,:,2][np.newaxis,:,:]
        beta=fcMatrix[:,:,3][np.newaxis,:,:]
        gamma_low=fcMatrix[:,:,4][np.newaxis,:,:]
        gamma_high=fcMatrix[:,:,5][np.newaxis,:,:]
        continue
    delta=np.concatenate((delta,fcMatrix[:,:,0][np.newaxis,:,:]),axis=0)
    theta=np.concatenate((theta,fcMatrix[:,:,1][np.newaxis,:,:]),axis=0)
    alpha=np.concatenate((alpha,fcMatrix[:,:,2][np.newaxis,:,:]),axis=0)
    beta=np.concatenate((beta,fcMatrix[:,:,3][np.newaxis,:,:]),axis=0)
    gamma_low=np.concatenate((gamma_low,fcMatrix[:,:,4][np.newaxis,:,:]),axis=0)
    gamma_high=np.concatenate((gamma_high,fcMatrix[:,:,5][np.newaxis,:,:]),axis=0)
    
mean_alpha=np.nansum(alpha,axis=0)
figure()
sns.heatmap(mean_alpha,cmap='jet')
figure()
sns.heatmap(alpha[0,:,:],cmap='jet')
bands_name=[str(x[0]) for x in mat['Freqs'][:,0]]
bands_freq=np.array([np.array(x[0].split(',')).astype(int) for x in mat['Freqs'][:,1]])
ROIs=[str(x[0][0]) for x in mat['Rows']]

fcDiagMat=[]
for e,file in enumerate(FcFile):
    mat = scipy.io.loadmat(path2fc+'/'+file)
    fcMatrix=np.arctanh(knn_graph(mat['TF_Expand_Matrix_Sorted'],Nneighbours=67))
    fcDiag=create_Graphs_Disconnected(fcMatrix)
    fcDiagMat.append(fcDiag)
fcDiagMat=np.array(fcDiagMat)
    


# ToDo hacer una clase con los atributos: num_nodes, num_edges, average node degree, 
# ToDo Probar si añadiendo una tansformacion laplaciana, la clasificacion mejora
#%% Generate labels

# scores=scoreDf_noNan[scoreDf.columns[[1,3,4,5,6,7,8,9,10]]].to_numpy()
scores=scoreDf_noNan[scoreDf.columns[[1]]].to_numpy()
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler.fit(scores)
# labels=scaler.transform(scores)
labels= scores

#%% Prepare dataset

# psd_train,psd_test,anat_train,anat_test,delta_train,delta_test,\
#         theta_train,theta_test, alpha_train,alpha_test,\
#         beta_train, beta_test, gamma1_train,gamma1_test,\
#         gamma2_train,gamma2_test, y_train, y_test,\
#         idx_train, idx_test=Split_Parallel_NN(restStatePCA,anatPCA,delta,theta,
#                                               alpha, beta, gamma_low,gamma_high,
#                                               labels,.3,seed=None)


dataset=CustomDataset(restStatePCA, anatPCA,delta, theta, 
                      alpha, beta, gamma_low,gamma_high,
                      labels, transform=None)



#%% making sure the dataset is properlly constructed
# and that the dataloader returns the proper structure
dataloader=DataLoader(dataset=dataset,batch_size=200,shuffle=True,num_workers=2)
dataiter=iter(dataloader)
data=next(dataiter)
psd, anat, de, th,al, be, g1, g2, l= data
print(psd.shape, anat.shape, de.shape, th.shape,\
      al.shape, be.shape, g1.shape, g2.shape, l.shape)

#%% Define input/output sizes, batchsize and learning rate

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

batch_size=128
lr = .0001
num_epochs = 150
input_size_psd = restStatePCA.shape[1]
input_size_anat = anatPCA.shape[1]
input_size_de = delta.shape[1:3]
input_size_th = theta.shape[1:3]
input_size_al = alpha.shape[1:3]
input_size_be = beta.shape[1:3]
input_size_g1 = gamma_low.shape[1:3]
input_size_g2 = gamma_high.shape[1:3]
output_size = labels.shape[1]




#%% NN just for anat

iterations=3



NNPred_list_anat=[]
for i in tqdm(range (iterations)):
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,
                            shuffle=True,num_workers=2)

    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,
                            shuffle=False,num_workers=2)
    if 'model' in globals():
        model.apply(weight_reset)
    model = NeuralNet(input_size_anat, output_size ).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    
    #training loop
    
    n_total_steps = len(train_loader) # number of batches
    for epoch in range (num_epochs):
        for i, (psd, anat, de, th,al, be, g1, g2, target) in enumerate(train_loader):
            anat=anat.to(device)
            target=target.to(device)
            
            #forward 
            outputs=model(anat)
            loss = criterion(outputs,target)
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch+1) % 10==0:
            #     print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
                
    
    #testing and eval
    pred=[]
    with torch.no_grad():
       
        for psd, anat, de, th,al, be, g1, g2, target in test_loader:
            anat=anat.to(device)
            target=target.to(device)
            outputs=model(anat).to('cpu').numpy()
            
            pred.extend(outputs)
    pred=np.array(pred)
    y_test=test_dataset[:][-1].numpy()
    NNPred=plotPredictionsReg(pred.flatten(),y_test.flatten(),False)
    NNPred_list_anat.append(NNPred)


NNPred_list_psd=[]
for i in tqdm(range (iterations)):
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,
                            shuffle=True,num_workers=2)

    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,
                            shuffle=False,num_workers=2)
    if 'model' in globals():
        model.apply(weight_reset)
    model = NeuralNet(input_size_psd, output_size ).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    
    #training loop
    
    n_total_steps = len(train_loader) # number of batches
    for epoch in range (num_epochs):
        for i, (psd, anat, de, th,al, be, g1, g2, target) in enumerate(train_loader):
            psd=psd.to(device)
            target=target.to(device)
            
            #forward 
            outputs=model(psd)
            loss = criterion(outputs,target)
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch+1) % 10==0:
            #     print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
                
    
    #testing and eval
    pred=[]
    with torch.no_grad():
       
        for psd, anat, de, th,al, be, g1, g2, target in test_loader:
            psd=psd.to(device)
            target=target.to(device)
            outputs=model(psd).to('cpu').numpy()
            
            pred.extend(outputs)
    pred=np.array(pred)
    y_test=test_dataset[:][-1].numpy()
    NNPred=plotPredictionsReg(pred.flatten(),y_test.flatten(),False)
    NNPred_list_psd.append(NNPred)
    

NNPred_list_CustomModel_NoFc=[]


for i in tqdm(range (iterations)):
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,
                            shuffle=True,num_workers=2)

    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,
                            shuffle=False,num_workers=2)
    if 'model' in globals():
        model.apply(weight_reset)
    model = CustomModel_NoFc(input_size_psd,input_size_anat,output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    
    #training loop
    
    n_total_steps = len(train_loader) # number of batches
    for epoch in range (num_epochs):
        for i, (psd, anat, de, th,al, be, g1, g2, target) in enumerate(train_loader):
            psd=psd.to(device)
            anat=anat.to(device)
            target=target.to(device)
            
            #forward 
            outputs=model(psd,anat)
            loss = criterion(outputs,target)
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch+1) % 10==0:
            #     print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
                
    
    #testing and eval
    pred=[]
    with torch.no_grad():
       
        for psd, anat, de, th,al, be, g1, g2, target in test_loader:
            psd=psd.to(device)
            anat=anat.to(device)
            target=target.to(device)
            outputs=model(psd,anat).to('cpu').numpy()
            
            pred.extend(outputs)
    pred=np.array(pred)
    y_test=test_dataset[:][-1].numpy()
    NNPred=plotPredictionsReg(pred.flatten(),y_test.flatten(),False)
    NNPred_list_CustomModel_NoFc.append(NNPred)
    


#%% Dataset Graph
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.loader import DataLoader
def select_feat(psdFeat,bandOfInt,anatFeat):
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
        return np.log(psdPCA[:,(freqsCropped>=bandOfInt[0])&(freqsCropped<=bandOfInt[1]),:])
    else:
        return myReshape(anatFeat)

# features= select_feat(psd2use,bands_freq,anatPCA)
features_pca= select_feat_psdPca(restStatePCA, anatPCA)
# feat_train,feat_test,alpha_train, alpha_test, y_train,y_test= train_test_split(features[0],alpha,age,test_size=.3)
feat_train,feat_test,alpha_train, alpha_test, y_train,y_test= train_test_split(features_pca,alpha,age,test_size=.3)
dataloader_train=DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train),batch_size=1,shuffle=True)
dataloader_test=DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test),batch_size=1)
dataset_test=Dataset_graph(feat_test, ROIs, alpha_test, y_test)
data=dataset_test[0]
print(data.y.shape)
# vis = to_networkx(data)
# node_labels = data.y.numpy()
# plt.figure(1,figsize=(15,13))
# nx.draw(vis, cmap=plt.get_cmap('Set3'),node_size=70, linewidths=6)

for tdata in dataloader_train:
    print(tdata)
    
    break

#%%
if 'model' in globals():
        model.apply(weight_reset)
model = GCN(data.x.shape[1],hidden_channels=12)
print(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=0.0001)
criterion = nn.MSELoss()    

def train(loader,epoch):
    model.train()#More details: model.train() sets the mode to train. You can call either model.eval() or model.train(mode=False) to tell that you are testing. It is somewhat intuitive to expect train function to train model but it does not do that. It just sets the mode.
    n_total_steps = len(dataloader_train) # number of batches

    for data in loader:  # Iterate in batches over the training dataset.
        data=data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass. Intenta agregar edge_attr
        loss = criterion(out, data.y)  # Compute the loss.
        optimizer.zero_grad()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    if (epoch+1) % 10==0:
        print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
    
        # optimizer.zero_grad()  # Clear gradients.
def test(loader,plot):
    model.eval()
    pred=[]
    true_label=[]
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            data=data.to(device)
            out = model(data.x, data.edge_index, data.batch).to('cpu').numpy()
            pred.extend(out)
            true_label.extend(data.y.to('cpu').numpy())
        pred=np.array(pred)
        true_label=np.array(true_label)
        NNPred=plotPredictionsReg(pred.flatten(),true_label.flatten(),plot)# Check against ground-truth labels.
        
    return pred,true_label,NNPred  # Derive ratio of correct predictions.

NNPred_list_CustomModel_Fc=[]
for i in range (iterations):
    feat_train,feat_test,alpha_train, alpha_test, y_train,y_test= train_test_split(features_pca,alpha,age,test_size=.3)
    dataloader_train=DataLoader(Dataset_graph(feat_train, ROIs, alpha_train, y_train),batch_size=1,shuffle=True)
    dataloader_test=DataLoader(Dataset_graph(feat_test, ROIs, alpha_test, y_test),batch_size=1)
    # data=dataset_test[0]
    del model
    if 'model' in globals():
            model.apply(weight_reset)
    
    model = GCN(data.x.shape[1],hidden_channels=12)
    # # print(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    criterion = nn.MSELoss()  
    
    for epoch in range(1, 150):
        train(dataloader_train,epoch)
        # _,_,train_acc = test(dataloader_train,False)
        # pred,true_label,test_acc = test(dataloader_test,False)
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    _,_,train_acc = test(dataloader_train,False)
    pred,true_label,test_acc = test(dataloader_test,False)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    NNPred_list_CustomModel_Fc.append(test_acc)

PredExperimentsDF=pd.DataFrame({'PSD_PCA':NNPred_list_psd,
                                'Anat_PCA':NNPred_list_anat,
                                'Parallel':NNPred_list_CustomModel_NoFc,
                                'GCN': NNPred_list_CustomModel_Fc})

figure()
PredExperimentsDf_melted = PredExperimentsDF.reset_index().melt(id_vars='index')
sns.boxplot(PredExperimentsDf_melted,y='value',x='variable')

title('Boxplot Acc')

#%%
# from modelCopyFromTutorial import *
# filters=32
# num_layers=2
# model_test = ChebNet(12, filters, 1, gcn_layer=num_layers,dropout=0.25,gcn_flag=True)
# #model_test = ChebNet(block_dura, filters, Nlabels, K=5,gcn_layer=num_layers,dropout=0.25)

# model_test = model_test.to(device)
# adj_mat = adj_mat.to(device)
# print(model_test)
# print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))

# optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)


#%% Dataset Graph_Heterogeneus

# from torch_geometric.data import DataLoader
# features= select_feat(psd2use,bands_freq,anatPCA)
# dataset_test=Dataset_graph(features[0], ROIs, alpha, age)
# data=dataset_test[0]
# vis = to_networkx(data)
# node_labels = data.y.numpy()
# plt.figure(1,figsize=(15,13))
# nx.draw(vis, cmap=plt.get_cmap('Set3'),node_size=70, linewidths=6)

# train_dataset, test_dataset = random_split(dataset_test, [train_size, test_size])
# train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,
#                         shuffle=True,num_workers=2)

# test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,
#                         shuffle=False,num_workers=2)

# dataiter=iter(train_loader)
# data=next(dataiter)
# print(data)