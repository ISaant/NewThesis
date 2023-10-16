#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 23:50:19 2023

@author: isaac
"""

import os
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
import pickle
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import tensorflow as tf
import scipy
from tqdm import tqdm
from matplotlib.pyplot import plot, figure, title
from Fun4newThesis import *
from FunClassifiers4newThesis import *
from sklearn.linear_model import Lasso
from copy import copy
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
AnatFile=np.sort(os.listdir(path2anat))
FcFile=np.sort(os.listdir(path2fc))
mainDir_psd=np.sort(os.listdir(path2psd))
emptyRoomDir=np.sort(os.listdir(path2psd+mainDir_psd[0]+'/'))
restStateDir=np.sort(os.listdir(path2psd+mainDir_psd[2]+'/'))



#%% Find nan values in the score dataframe
with open(current_path+'/scoreDf.pickle', 'rb') as f:
    scoreDf = pickle.load(f)

#lets just keep age for now:
scoreDf.drop(columns=['Acer','BentonFaces','Cattell','EmotionRecog','Hotel','Ppp','Synsem','VSTM'],inplace=True)
row_idx=np.where(np.isnan(scoreDf.iloc[:,3:-1].to_numpy()))[0] #rows where there is nan

scoreDf_noNan=scoreDf.drop(row_idx)
#%% Demographics
demographics=pd.read_csv(path2psd+mainDir_psd[1])
subjects=demographics['CCID']

#%% Read PSD

for e,file in enumerate(tqdm(restStateDir)):
    matrix=myReshape(pd.read_csv(path2psd+mainDir_psd[2]+'/'+file,header=None).to_numpy())[np.newaxis, :]
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
DataScaled=anat2use
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
sns.kdeplot(MatPredDf_melted,x='value',hue='variable',fill=True, 
            common_norm=False, palette="rainbow",alpha=.5, linewidth=1)

title('Feature importance, "Flip" approach')

anat2use=RestoreShape(np.delete(myReshape(anat2use),[0,1,7],axis=1))
# Rearange for NN
cont=0
Anat_aranged=np.zeros((anat2use.shape)) #!!! YA NO PUEDES RESTAURAR A (SUB,ANAT,ROI) USANDO RESTORESHAPE
for i in range (6):
    for j in np.arange(0,6*68,6):
        print(i+j)
        Anat_aranged[:,cont]=anat2use[:,i+j]
        cont+=1

#%% Read Fc
# Check if the order is the same

boolarray=[x[4:-4]==y[4:] for x,y in zip(FcFile,subjects) ]

print('All the subjects are sorted equal between the datasets: '+str(any(boolarray)) )

for e,file in enumerate(FcFile):
    mat = scipy.io.loadmat(path2fc+'/'+file)
    fcMatrix=mat['TF_Expand_Matrix_Sorted']
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
bands=[str(x[0]) for x in mat['Freqs'][:,0]]
ROIs=[str(x[0][0]) for x in mat['Rows']]

#%% Generate labels

# scores=scoreDf_noNan[scoreDf.columns[[1,3,4,5,6,7,8,9,10]]].to_numpy()
scores=scoreDf_noNan[scoreDf.columns[[1]]].to_numpy().flatten()
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler.fit(scores)
# labels=scaler.transform(scores)
labels= scores
#%% individual NN psd_PCA

# DataScaled=Scale(restStatePCA)
NNPred_list_psd=[]
for i in tqdm(range(100)):

    x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(restStatePCA,labels,.3)
    Input0=tf.keras.Input(shape=(x_train.shape[1],), )
    modelNN=Perceptron_PCA (Input0,1)
    trainModel(modelNN,x_train,y_train,150,False)
    # predNN=evaluateRegModel(model,x_test,y_test)
    predNN = modelNN.predict(x_test)
    NNPred=np.empty(predNN.shape[1])
    # for i in range(predNN.shape[1]):
        # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
    NNPred=plotPredictionsReg(predNN.flatten(),y_test,False)
    NNPred_list_psd.append(NNPred)
    
# individual NN anat

# NNPred_list_anat_Aranged=[]
# for i in tqdm(range(100)):

#     x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(Anat_aranged,labels,.3)
#     x_train=Scale(x_train)
#     x_test=Scale(x_test)
#     Input0=tf.keras.Input(shape=(x_train.shape[1],), )
#     modelNN=Perceptron_Anat (Input0,1)
#     trainModel(modelNN,x_train,y_train,150,False)
#     # predNN=evaluateRegModel(model,x_test,y_test)
#     predNN = modelNN.predict(x_test)
#     NNPred=np.empty(predNN.shape[1])
#     # for i in range(predNN.shape[1]):
#         # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
#     NNPred=plotPredictionsReg(predNN.flatten(),y_test,False)
#     # print(NNPred)
#     NNPred_list_anat_Aranged.append(NNPred)


NNPred_list_anat_PCA=[]
for i in tqdm(range(100)):

    x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(anatPCA,labels,.3)
    x_train=Scale(x_train)
    x_test=Scale(x_test)
    Input0=tf.keras.Input(shape=(x_train.shape[1],), )
    modelNN=Perceptron_Anat (Input0,1)
    trainModel(modelNN,x_train,y_train,150,False)
    # predNN=evaluateRegModel(model,x_test,y_test)
    predNN = modelNN.predict(x_test)
    NNPred=np.empty(predNN.shape[1])
    # for i in range(predNN.shape[1]):
        # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
    NNPred=plotPredictionsReg(predNN.flatten(),y_test,False)
    # print(NNPred)
    NNPred_list_anat_PCA.append(NNPred)

# NNPred_list_anat_NoAranged=[]
# for i in tqdm(range(100)):

#     x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(anat2use,labels,.3)
#     x_train=Scale(x_train)
#     x_test=Scale(x_test)
#     Input0=tf.keras.Input(shape=(x_train.shape[1],), )
#     modelNN=Perceptron_Anat (Input0,1)
#     trainModel(modelNN,x_train,y_train,150,False)
#     # predNN=evaluateRegModel(model,x_test,y_test)
#     predNN = modelNN.predict(x_test)
#     NNPred=np.empty(predNN.shape[1])
#     # for i in range(predNN.shape[1]):
#         # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
#     NNPred=plotPredictionsReg(predNN.flatten(),y_test,False)
#     # print(NNPred)
#     NNPred_list_anat_NoAranged.append(NNPred)
# # #%% Parallel NN

# anat2use=np.delete(AnatomicalFeatures, row_idx,axis=0)
# NNPred_list_anat_Original=[]
# for i in tqdm(range(100)):

#     x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(anat2use,labels,.3)
#     x_train=Scale(x_train)
#     x_test=Scale(x_test)
#     Input0=tf.keras.Input(shape=(x_train.shape[1],), )
#     modelNN=Perceptron_Anat (Input0,1)
#     trainModel(modelNN,x_train,y_train,150,False)
#     # predNN=evaluateRegModel(model,x_test,y_test)
#     predNN = modelNN.predict(x_test)
#     NNPred=np.empty(predNN.shape[1])
#     # for i in range(predNN.shape[1]):
#         # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
#     NNPred=plotPredictionsReg(predNN.flatten(),y_test,False)
#     # print(NNPred)
#     NNPred_list_anat_Original.append(NNPred)



# Create parallel NN


NNPred_list_ParallelNN=[]
for i in range(100):

    psd_train, psd_test, anat_train, anat_test, y_train, y_test = train_test_split(restStatePCA,
                                                                    anatPCA,
                                                                    labels,test_size=.3)
    
    psd_train=Scale(psd_train)
    psd_test=Scale(psd_test)
    anat_train=Scale(anat_train)
    anat_test=Scale(anat_test)
    
    InputPSD=tf.keras.Input(shape=(psd_train.shape[1],), )
    InputAnat=tf.keras.Input(shape=(anat_train.shape[1],), )
    
    modelNN=parallelNN (InputPSD,InputAnat,1)
    trainModel(modelNN,[psd_train,anat_train],y_train,200,False)
    # predNN=evaluateRegModel(model,x_test,y_test)
    predNN = modelNN.predict([psd_test,anat_test])
    # NNPred=np.empty(predNN.shape[1])
    # for i in range(predNN.shape[1]):
        # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
    NNPred=plotPredictionsReg(predNN.flatten(),y_test,False)
    NNPred_list_ParallelNN.append(NNPred)
    # print(NNPred)

PredExperimentsDF=pd.DataFrame({'PSD_PCA':NNPred_list_psd,
                                'Anat_PCA':NNPred_list_anat_PCA,
                                'Parallel':NNPred_list_ParallelNN })

PredExperimentsDf_melted = PredExperimentsDF.reset_index().melt(id_vars='index')
sns.boxplot(PredExperimentsDf_melted,y='value',x='variable')

title('Boxplot Acc anat and PSD')

#%% Dataset Graph

freqs2use
features = np.concatenate((np.log(restState),myReshape(anatPCA)),axis=1)
