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
from netneurotools.networks import threshold_network as threshNet

def read_Fc(FcFile,path,subjects, per=40):

    
    '''Aqui debes decidir si usar k vecinos o umbralizar. knn nos dara una matriz sparce uniforme'''
    cont = 0
    for file in tqdm(FcFile):
        if len(np.argwhere(subjects == file[:-7])) == 0:
            continue
        mat = scipy.io.loadmat(path+'/'+file)
        # fcMatrix=knn_graph(mat['TF_Expand_Matrix_Sorted'],Nneighbours=67)
        # fcMatrix=np.arctanh(threshold(mat['TF_Expand_Matrix_Sorted'],tresh=.4))
        # fcMatrix=percentage(mat['TF_Expand_Matrix_Sorted'],per=per)
        fcMatrix = mat['TF_Expand_Matrix_Sorted']
        if cont==0:
            delta=(fcMatrix[:,:,0]*threshNet(fcMatrix[:,:,0],per))[np.newaxis,:,:]
            theta=(fcMatrix[:,:,1]*threshNet(fcMatrix[:,:,1],per))[np.newaxis,:,:]
            alpha=(fcMatrix[:,:,2]*threshNet(fcMatrix[:,:,2],per))[np.newaxis,:,:]
            beta=(fcMatrix[:,:,3]*threshNet(fcMatrix[:,:,3],per))[np.newaxis,:,:]
            gamma_low=(fcMatrix[:,:,4]*threshNet(fcMatrix[:,:,4],per))[np.newaxis,:,:]
            gamma_high=(fcMatrix[:,:,5]*threshNet(fcMatrix[:,:,5],per))[np.newaxis,:,:]
            cont += 1
            continue
        delta=np.concatenate((delta,(fcMatrix[:,:,0]*threshNet(fcMatrix[:,:,0],per))[np.newaxis,:,:]),axis=0)
        theta=np.concatenate((theta,(fcMatrix[:,:,1]*threshNet(fcMatrix[:,:,1],per))[np.newaxis,:,:]),axis=0)
        alpha=np.concatenate((alpha,(fcMatrix[:,:,2]*threshNet(fcMatrix[:,:,2],per))[np.newaxis,:,:]),axis=0)
        beta=np.concatenate((beta,(fcMatrix[:,:,3]*threshNet(fcMatrix[:,:,3],per))[np.newaxis,:,:]),axis=0)
        gamma_low=np.concatenate((gamma_low,(fcMatrix[:,:,4]*threshNet(fcMatrix[:,:,4],per))[np.newaxis,:,:]),axis=0)
        gamma_high=np.concatenate((gamma_high,(fcMatrix[:,:,5]*threshNet(fcMatrix[:,:,5],per))[np.newaxis,:,:]),axis=0)
        cont += 1
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma_low', 'gamma_high']



    connectomes = dict.fromkeys(band_names, [])
    for band, band_mat in zip(band_names,[delta, theta, alpha, beta, gamma_low, gamma_high]):
        connectomes[band]= band_mat
    # mean_delta=np.nansum(delta,axis=0)
    # mean_theta=np.nansum(theta,axis=0)
    # mean_alpha=np.nansum(alpha,axis=0)
    # mean_beta=np.nansum(beta,axis=0)
    # mean_gamma_low=np.nansum(gamma_low,axis=0)
    # mean_gamma_high=np.nansum(gamma_high,axis=0)
    # plt.figure()
    # sns.heatmap(mean_delta,cmap='jet')
    # plt.figure()
    # sns.heatmap(mean_theta,cmap='jet')
    # plt.figure()
    # sns.heatmap(mean_alpha,cmap='jet')
    # plt.figure()
    # sns.heatmap(mean_beta,cmap='jet')
    # plt.figure()
    # sns.heatmap(mean_gamma_low,cmap='jet')
    # plt.figure()
    # sns.heatmap(mean_gamma_high,cmap='jet')
    # plt.figure()
    # bands_name=[str(x[0]) for x in mat['Freqs'][:,0]]
    # bands_freq=np.array([np.array(x[0].split(',')).astype(int) for x in mat['Freqs'][:,1]])
    ROIs=[str(x[0][0]) for x in mat['Rows']]
    
    # fcDiagMat=[]
    # for e,file in enumerate(FcFile): #Aqui unimos todos los conectomas en un solo grafo diagonalizado deconexo, la intencion es hacer pruebas mas adelante
    #     mat = scipy.io.loadmat(path+'/'+file)
    #     fcMatrix=np.arctanh(knn_graph(mat['TF_Expand_Matrix_Sorted'],Nneighbours=67))
    #     fcDiag=create_Graphs_Disconnected(fcMatrix)
    #     fcDiagMat.append(fcDiag)
    # fcDiagMat=np.array(fcDiagMat)


    return connectomes, ROIs