#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 23:50:19 2023

@author: isaac
"""

import os
os.chdir('/home/isaac/Documents/Doctorado_CIC/Internship/Sylvain/New_thesis/Python_Fun/')
import pickle
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import tensorflow as tf
import pickle
import math
import scipy
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from Fun4newThesis import *
from FunClassifiers4newThesis import *
#%% Hyperparameters

#PSD
freqs=np.arange(0,150,.5)
freqs2use=[0,100]
columns= [i for i, x in enumerate((freqs>=freqs2use[0]) & (freqs<freqs2use[1])) if x]
freqsCropped=freqs[columns]
#columns is used to select the region of the PSD we are interested in


#%%  Directories
current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path,'../../'))
path2psd = parentPath+'/Stability-project/Stability-project_db/CAMCAN_Jason_PrePro/'
path2anat =  parentPath+'/Stability-project/Stability-project_db/Anatomical_Features/'
path2fc= parentPath+'/New_thesis/camcan_AEC_ortho_Matrix'
AnatFile=np.sort(os.listdir(path2anat))
FcFile=np.sort(os.listdir(path2fc))
mainDir_psd=np.sort(os.listdir(path2psd))
emptyRoomDir=np.sort(os.listdir(path2psd+mainDir_psd[0]+'/'))
restStateDir=np.sort(os.listdir(path2psd+mainDir_psd[2]+'/'))
taskDir=np.sort(os.listdir(path2psd+mainDir_psd[3]+'/'))


#%% Find nan values in the score dataframe
with open(current_path+'/scoreDf.pickle', 'rb') as f:
    scoreDf = pickle.load(f)

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

#%% Read Anat

CorticalThickness=pd.read_csv(path2anat+AnatFile[0],header=None)
anat2use=np.delete(CorticalThickness, row_idx,axis=0)

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
bands=bands=[str(x[0]) for x in mat['Freqs'][:,0]]
ROIs=[str(x[0][0]) for x in mat['Rows']]

#%% Generate labels

scores=scoreDf_noNan[scoreDf.columns[[1,3,4,5,6,7,8,9,10]]].to_numpy()
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(scores)
# labels=scaler.transform(scores)
labels= scores
#%% individual NN

# DataScaled=Scale(restStatePCA)

for i in range(10):

    x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(restStatePCA,labels[:,i],.3)
    Input0=tf.keras.Input(shape=(x_train.shape[1],), )
    modelNN=Perceptron_PCA (Input0,1)
    trainModel(modelNN,x_train,y_train,500,False)
    # predNN=evaluateRegModel(model,x_test,y_test)
    predNN = modelNN.predict(x_test)
    NNPred=np.empty(predNN.shape[1])
    # for i in range(predNN.shape[1]):
        # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
    NNPred=plotPredictionsReg(predNN.flatten(),y_test,True)
    print(NNPred)

#%% Parallel NN


# DataScaled=Scale(restStatePCA)

for i in range(10):

    psd_train, psd_test, anat_train, anat_test, y_train, y_test = train_test_split(restStatePCA,
                                                                   anat2use,
                                                                   labels[:,i],test_size=.3)
    modelNN=parallelNN (psd_train,anat_train,_,1)
    trainModel(modelNN,[psd_train,anat_train],y_train,200,False)
    # predNN=evaluateRegModel(model,x_test,y_test)
    predNN = modelNN.predict([psd_test,anat_test])
    # NNPred=np.empty(predNN.shape[1])
    # for i in range(predNN.shape[1]):
        # NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
    NNPred=plotPredictionsReg(predNN.flatten(),y_test,True)
    print(NNPred)

#%% All scores one feature
x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(restStatePCA,scores,.3)
Input0=tf.keras.Input(shape=(x_train.shape[1],), )
modelNN=Perceptron (Input0,9)
trainModel(modelNN,x_train,y_train,500,True)
# predNN=evaluateRegModel(model,x_test,y_test)
predNN = modelNN.predict(x_test)
NNPred=np.empty(predNN.shape[1])
for i in range(predNN.shape[1]):
    NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
# NNPred=plotPredictionsReg(predNN.flatten(),y_test,True)
print(NNPred)

#%% All scores all features

# DataScaled=Scale(restStatePCA)


psd_train, psd_test, anat_train, anat_test, y_train, y_test = train_test_split(restStatePCA,
                                                               anat2use,
                                                               labels,test_size=.3)
modelNN=parallelNN (psd_train,anat_train,_,9)
trainModel(modelNN,[psd_train,anat_train],y_train,500,True)
# predNN=evaluateRegModel(model,x_test,y_test)
predNN = modelNN.predict([psd_test,anat_test])
NNPred=np.empty(predNN.shape[1])
for i in range(predNN.shape[1]):
    NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
print(NNPred)

#%% All scores one feature NN2.0
x_train, x_test, y_train,y_test,idxTrain_psd,idxTest_psd=Split(restStatePCA,scores,.3)
Input0=tf.keras.Input(shape=(x_train.shape[1],), )
modelNN=Perceptron (Input0,9)
trainModel(modelNN,x_train,y_train,500,True)
# predNN=evaluateRegModel(model,x_test,y_test)
predNN = modelNN.predict(x_test)
NNPred=np.empty(predNN.shape[1])
for i in range(predNN.shape[1]):
    NNPred[i]=plotPredictionsReg(predNN[:,i],y_test[:,i],True)
# NNPred=plotPredictionsReg(predNN.flatten(),y_test,True)
print(NNPred)

#%% All scores all features NN2.0

# DataScaled=Scale(restStatePCA)


psd_train, psd_test, anat_train, anat_test, y_train, y_test = train_test_split(restStatePCA,
                                                               anat2use,
                                                               labels,test_size=.3)
modelNN=parallelNN2p0 (psd_train,anat_train,_)
trainModel(modelNN,[psd_train,anat_train],y_train,500,True)
# predNN=evaluateRegModel(model,x_test,y_test)
predNN = modelNN.predict([psd_test,anat_test])
NNPred=np.empty(y_test.shape[1])
for i in range(y_test.shape[1]):
    NNPred[i]=plotPredictionsReg(predNN[i].flatten(),y_test[:,i],False)
print(NNPred)