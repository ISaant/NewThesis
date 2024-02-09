#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:03:35 2023

@author: isaac
"""

import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
from Generate_Features_Dataloades import Generate
from importlib import reload
from Fun4newThesis import *
from FunClassifiers4newThesis_pytorch import *
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

psd2use, restStatePCA, anat2use, anatPCA, DiagFc, restStatePCA_s200, anatPCA_s200, structConn, local, glob, ROIs, scores = Generate()

#%% Prepare dataset

print('Generating Dataset for traditional ML...')
# scores_scaled,scaler=Scale_out(scores)
dataset=CustomDataset(restStatePCA, anatPCA, DiagFc, restStatePCA_s200, anatPCA_s200, np.concatenate((np.array(local), glob),axis=1), scores, transform=None)


def dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=[]):
    gen = torch.Generator()
    if seed:
        gen.manual_seed(seed)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size], generator = gen)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                             shuffle=False, num_workers=2)

    val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset),
                            shuffle=False, num_workers=2)

    return train_loader, test_loader, val_loader

dataloader=DataLoader(dataset=dataset,batch_size=200,shuffle=True,num_workers=2)
dataiter=iter(dataloader)
data=next(dataiter)
#----------------------#
#For indexing dataset/dataiter: 0:psd,1:anat,2:embeddings,3:tagets
#----------------------#

psd,anat, fc, psds200, anats200, ScFeat, l= data
print(psd.shape ,anat.shape , fc.shape, psds200.shape, anats200.shape, ScFeat.shape, l.shape)

# Define input/output sizes, batchsize and learning rate

train_size = int(0.7 * len(dataset))
test_size = int(0.9 * len(dataset)) - train_size
val_size = len(dataset) - train_size - test_size

print(train_size+test_size+val_size == len(dataset))

batch_size=128 #Asegurate de que sea mayor al tamaño del test
lr = .0001
num_epochs = 150
input_size_psd = psd.shape[1]
input_size_anat = anat.shape[1]
input_size_fc = fc.shape[1]
input_size_psds200 = psds200.shape[1]
input_size_anats200 = anats200.shape[1]
input_size_ScFeat = ScFeat.shape[1]
# input_size_global_sc = glob_sc.shape[1]
output_size = scores.shape[1]


#%% Prepare dataset for GCN
from torch_geometric.loader import DataLoader as DataLoader_gometric
print ('Generating Dataset for GCN with Structural connectivity...')

def select_feat_psdPca(psdPCA,anatFeat):
    psdPCA = myReshape(psdPCA,rois=200)
    anatFeat = myReshape(anatFeat,rois=200)
    # anatFeat=np.tile(anatFeat,numBandsUsed)
    if len(psdPCA)!=0 and len(anatFeat)!=0:

        featlist=list(np.concatenate((psdPCA,anatFeat),axis=1))
        
        return featlist
    elif len(psdPCA)!=0 and len(anatFeat)==0:
        # return psdPCA[:,(freqsCropped>=bandOfInt[0])&(freqsCropped<=bandOfInt[1]),:]
        return list(psdPCA)
    else:
        return list(myReshape(anatFeat))
    
features_pca= select_feat_psdPca(restStatePCA_s200, anatPCA_s200)

feat_train, feat_test, sc_train, sc_test, y_train, y_test = train_test_split(
    features_pca, structConn, scores, test_size=.2,random_state=12)

feat_train, feat_val, sc_train, sc_val, y_train, y_val = train_test_split(
    feat_train, sc_train, y_train, test_size=.1, random_state=12)

dataloader_train = DataLoader_gometric(Dataset_graph(feat_train, ROIs, sc_train, y_train), batch_size=1,
                              shuffle=True)
dataloader_test = DataLoader_gometric(Dataset_graph(feat_test, ROIs, sc_test, y_test), batch_size=1)

dataloader_val = DataLoader_gometric(Dataset_graph(feat_val, ROIs, sc_val, y_val), batch_size=1)

dataset_all=Dataset_graph(features_pca, ROIs, structConn, scores)

for i in range(10):
    data = dataset_all[i]
    print(data)


#%%
iterations = 30
model = []
import FunClassifiers4newThesis_pytorch
reload(FunClassifiers4newThesis_pytorch)
from FunClassifiers4newThesis_pytorch import *

#%% Lasso For Anat
#


Mean_Acc=[]
ROIcsv=pd.read_csv('~/Documents/Doctorado_CIC/NewThesis/dk_ROI_Importance.csv')
for i in range(50):
    anat_train, anat_test, y_train, y_test, _, _ = Split(anatPCA, scores, .3)
    anat_train = Scale(anat_train)
    anat_test = Scale(anat_test)
    model = Lasso(alpha=.2)
    model.fit(anat_train, y_train)
    pred_Lasso = model.predict(anat_test)
    LassoPred,LassoMAE = plotPredictionsReg(pred_Lasso, y_test.flatten(), False)
    Mean_Acc.append(LassoPred)
Mean_Acc=np.mean(Mean_Acc)

anatPCA = myReshape(anatPCA)
Sub,Feat,ROI = anatPCA.shape
ROI_Importance = []
for roi in tqdm(range(ROI)):
    reg_roi=[]
    for i in range(50):
        anat_train, anat_test, y_train,y_test,_,_=Split(anatPCA[:,:,roi], scores,.3)
        anat_train = Scale(anat_train)
        anat_test = Scale(anat_test)
        model = Lasso(alpha=.2)
        model.fit(anat_train, y_train)
        pred_Lasso = model.predict(anat_test)
        LassoPred,LassoMAE = plotPredictionsReg(pred_Lasso, y_test.flatten(), False)
        reg_roi.append(LassoPred)
    ROI_Importance.append(np.mean(reg_roi))
ROI_Importance = np.take(ROI_Importance,np.argsort(ROIs))
# ROI_Importance = Mean_Acc*(1-(Mean_Acc-ROI_Importance))

ROIcsv['Acc'] = ROI_Importance
ROIcsv.to_csv('~/Documents/Doctorado_CIC/NewThesis/dk_ROI_Importance.csv',index=False)

sns.barplot(ROIcsv,x='network', y = 'Acc', palette='mako')
plt.xticks(rotation = 45)
plt.ylim(.3,.71)
anatPCA = RestoreShape(anatPCA)
#%% Individual NN
#psd
NNPred_psd=[]
MAE_psd=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_psd, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 0
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred,NNMAE = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_psd.append(NNPred)
    MAE_psd.append(NNMAE)
#%%anat
NNPred_anat=[]
MAE_anat=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_anat, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 1
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred,NNMAE = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_anat.append(NNPred)
    MAE_anat.append(NNMAE)

#%%psd_s200
NNPred_psd_s200=[]
MAE_psd_s200=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_psds200, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 3
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred,NNMAE = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_psd_s200.append(NNPred)
    MAE_psd_s200.append(NNMAE)
#%%anat_s200
NNPred_anat_s200=[]
MAE_anat_s200=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = NeuralNet(input_size_anats200, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    # Puedes imprimir el resumen del modelo si lo deseas
    # print(model)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = 4
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    # pred = scaler.inverse_transform(np.array(pred).reshape(-1,1))
    # y_test = scaler.inverse_transform(test_data[:][-1].numpy())
    # NNPred = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred,NNMAE = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_anat_s200.append(NNPred)
    MAE_anat_s200.append(NNMAE)
#%% Parallel NN
NNPred_CustomModel=[]
MAE_CustomModel=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = CustomModel_NoFc(input_size_psd, input_size_anat, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = [0, 1]
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred,NNMAE = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_CustomModel.append(NNPred)
    MAE_CustomModel.append(NNMAE)
#%%
NNPred_CustomModel_s200=[]
MAE_CustomModel_s200=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=i)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = CustomModel_NoFc(input_size_psds200, input_size_anats200, output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = [3, 4]
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred,NNMAE = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_CustomModel_s200.append(NNPred)
    MAE_CustomModel_s200.append(NNMAE)

#%%
NNPred_CustomModel_Sc=[]
MAE_CustomModel_Sc=[]
for i in tqdm(range(iterations)):
    train_loader, test_loader, val_loader = dataloaders(dataset, train_size, test_size, val_size, batch_size, seed=1)
    dataiter = iter(test_loader)
    test_data = next(dataiter)
    model = CustomModel_Sc(input_size_psds200, input_size_anats200, input_size_ScFeat,  output_size).to(device)
    if 'model' in globals():
        model.apply(weight_reset)
    print(model._get_name())
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    var_pos = [3, 4, 5]
    train_ANN(model, criterion, optimizer, train_loader, val_loader, num_epochs, var_pos)
    mse_test, pred = test_ANN(model, test_loader, var_pos)
    pred = np.array(pred)
    y_test = test_data[:][-1].numpy()
    NNPred,NNMAE = plotPredictionsReg(pred.flatten(), y_test.flatten(), False)
    print(f'Test_Acc: {NNPred:4f}, Test_MSE: {mse_test}')
    NNPred_CustomModel_Sc.append(NNPred)
    MAE_CustomModel_Sc.append(NNMAE)
#%%
PredExperimentsDF=pd.DataFrame({'PSD_PCA':list(np.array(NNPred_psd_s200)),
                                'Anat_PCA':list(np.array(NNPred_anat_s200)),
                                'Parallel_noSc':list(np.array(NNPred_CustomModel_s200)),
                                'Parallel_Sc':list(np.array(NNPred_CustomModel_Sc)),
                                #'GCN': models_acc[0],
                                #'SAGE_GCN': models_acc[1],
                                #'GNN_Diffpool': models_acc[2],
                                #'GIN':NNPred_list_CustomModel_Fc
                                })

plt.figure()
PredExperimentsDf_melted = PredExperimentsDF.reset_index().melt(id_vars='index')
sns.boxplot(PredExperimentsDf_melted,y='value',x='variable', palette="dark:#5A9_r")
plt.xlabel('Model')
plt.ylabel('Performance')
plt.title('Boxplot Acc')

#%%
MAEExperimentsDF=pd.DataFrame({'PSD_PCA':list(np.array(MAE_psd_s200)),
                                'Anat_PCA':list(np.array(MAE_anat_s200)),
                                'Parallel_noSc':list(np.array(MAE_CustomModel_s200)),
                                'Parallel_Sc':list(np.array(MAE_CustomModel_Sc)),
                                #'GCN': models_acc[0],
                                #'SAGE_GCN': models_acc[1],
                                #'GNN_Diffpool': models_acc[2],
                                #'GIN':NNPred_list_CustomModel_Fc
                                })

plt.figure()
MAEExperimentsDf_melted = MAEExperimentsDF.reset_index().melt(id_vars='index')
sns.boxplot(MAEExperimentsDf_melted,y='value',x='variable', palette="dark:#5A9_r")
plt.xlabel('Model')
plt.ylabel('Performance')
plt.title('Boxplot Acc')

#%%
model1 = GCN(data.x.shape[1],hidden_channels=12, lin=True)
model2 = SAGE_GCN (data.x.shape[1],hidden_channels=32, lin=True)
model3 = GNN_DiffPool(data.x.shape[1])

models = [model1,model2,model3]
models_acc=[]
num_epochs= 200
models_loss= np.zeros((3,num_epochs))

for i in tqdm(range(3)):

    feat_train, feat_test, sc_train, sc_test, y_train, y_test = train_test_split(
        features_pca, structConn, scores, test_size=.2,random_state=12)

    feat_train, feat_val, sc_train, sc_val, y_train, y_val = train_test_split(
        feat_train, sc_train, y_train, test_size=.1, random_state=12)

    dataloader_train = DataLoader_gometric(Dataset_graph(feat_train, ROIs, sc_train, y_train), batch_size=1,
                                  shuffle=True)
    dataloader_test = DataLoader_gometric(Dataset_graph(feat_test, ROIs, sc_test, y_test), batch_size=1)

    dataloader_val = DataLoader_gometric(Dataset_graph(feat_val, ROIs, sc_val, y_val), batch_size=1)
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
