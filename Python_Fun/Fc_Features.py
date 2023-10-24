#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:50:27 2023

@author: sflores
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Fun4newThesis import *
from tqdm import tqdm
from FunClassifiers4newThesis_pytorch import *

def Fc_Feat(FcFile,path):

    
    '''Aqui debes decidir si usar k vecinos o umbralizar. knn nos dara una matriz sparce uniforme'''
    for e,file in enumerate(tqdm(FcFile)):
        mat = scipy.io.loadmat(path+'/'+file)
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
    plt.figure()
    sns.heatmap(mean_alpha,cmap='jet')
    plt.figure()
    sns.heatmap(alpha[0,:,:],cmap='jet')
    bands_name=[str(x[0]) for x in mat['Freqs'][:,0]]
    bands_freq=np.array([np.array(x[0].split(',')).astype(int) for x in mat['Freqs'][:,1]])
    ROIs=[str(x[0][0]) for x in mat['Rows']]
    
    fcDiagMat=[]
    for e,file in enumerate(FcFile): #Aqui unimos todos los conectomas en un solo grafo diagonalizado deconexo, la intencion es hacer pruebas mas adelante
        mat = scipy.io.loadmat(path+'/'+file)
        fcMatrix=np.arctanh(knn_graph(mat['TF_Expand_Matrix_Sorted'],Nneighbours=67))
        fcDiag=create_Graphs_Disconnected(fcMatrix)
        fcDiagMat.append(fcDiag)
    fcDiagMat=np.array(fcDiagMat)
    
    return delta, theta, alpha, beta, gamma_low, gamma_high, ROIs