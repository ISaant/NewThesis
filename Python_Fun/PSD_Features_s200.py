#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:50:27 2023

@author: sflores
"""
import os

import numpy as np
import scipy
import matplotlib.pyplot as plt
from Fun4newThesis import RestoreShape, myPCA
from tqdm import tqdm
def PSD_Feat_s200 (path, PSDFile_s200, subjects):
        cont = 0
        for file in tqdm(PSDFile_s200):
                if len(np.argwhere(subjects == file[:-8]))==0:
                        continue
                mat = scipy.io.loadmat(path+'/'+file)
                # subject = str(mat['DataFile'])
                # subject = subject[subject.find('CC'):subject.find('.mat')]
                restState = np.squeeze(mat['TF']).T[np.newaxis, :, :]
                if cont == 0:
                        freqs = mat['Freqs'][0]
                        freqs2use = [0, 90]
                        columns = [i for i, x in enumerate((freqs >= freqs2use[0]) & (freqs < freqs2use[1])) if x]
                        restStateAll = restState
                else:
                        restStateAll = np.concatenate((restStateAll, restState))
                cont += 1
        # psd2use = np.delete(restStateAll, row_idx, axis=0)[:, columns,:]  # remove index with nan and select the band-width of interest
        freqs2use = [0, 90]
        freqs = mat['Freqs'].flatten()
        columns = [i for i, x in enumerate((freqs >= freqs2use[0]) & (freqs < freqs2use[1])) if x]
        psd2use = restStateAll[:,columns,:]
        # PCA on PSD
        Sub, PSD, ROI = psd2use.shape
        nPCA = 10
        restStatePCA = np.zeros((Sub, nPCA, ROI))
        for roi in range(ROI):
                pca_df, pca2use, prop_varianza_acum = myPCA(np.log(psd2use[:, :, roi]), False, nPCA)
                plt.plot(prop_varianza_acum[:10])
                restStatePCA[:, :, roi] = np.array(pca2use)

        restStatePCA = RestoreShape(restStatePCA)

        return psd2use, restStatePCA
                # os.rename(path+'/'+file, path+'/sub_'+subject+'_psd.mat')
                # if file[4:-8] == subject:
                #         print(True)
#%% This code will only be needed if you need to check if the repeated subjects share
# the same data. The do!

# IDs,count = np.unique(subjectsID, return_counts=True)
#
# repeated = np.argwhere(count>1).flatten()
#
# for rep in repeated:
#         posrep=np.argwhere(np.array(subjectsID) == IDs[rep]).flatten()
#         TF=[]
#         for pos in posrep:
#                 file = PSDFile_s200[pos]
#                 mat = scipy.io.loadmat(path + '/' + file)
#                 TF.append(np.squeeze(mat['TF']))
#
#
#         areEqual=np.where(TF[0] != TF[1])[0]
#         if len(areEqual) == 0:
#                 print(f' sub: {posrep[0]} and sub: {posrep[1]}, son iguales')



#%%

# TF=[]
# file = 'sub_CC410015_02_psd.mat'
# mat = scipy.io.loadmat(path + '/' + file)
# TF.append(np.squeeze(mat['TF']))
# file = 'sub_CC410015_psd.mat'
# mat = scipy.io.loadmat(path + '/' + file)
# TF.append(np.squeeze(mat['TF']))
# print(np.where(TF[0] != TF[1]))
# TF=[]
# file = 'sub_CC410040_02_psd.mat'
# mat = scipy.io.loadmat(path + '/' + file)
# TF.append(np.squeeze(mat['TF']))
# file = 'sub_CC410040_psd.mat'
# mat = scipy.io.loadmat(path + '/' + file)
# TF.append(np.squeeze(mat['TF']))
# print(np.where(TF[0] != TF[1]))
# TF=[]
# file = 'sub_CC410032_02_psd.mat'
# mat = scipy.io.loadmat(path + '/' + file)
# TF.append(np.squeeze(mat['TF']))
# file = 'sub_CC410032_psd.mat'
# mat = scipy.io.loadmat(path + '/' + file)
# TF.append(np.squeeze(mat['TF']))
# print(np.where(TF[0] != TF[1]))
#%%



#%%

