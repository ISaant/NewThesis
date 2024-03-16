#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:32:54 2023

@author: sflores
"""

import numpy as np
import pandas as pd
from Fun4newThesis import *
from tqdm import tqdm

def PSD_Feat (path,mainDir,restStateDir,emptyRoomDir,columns, row_idx, sort_idx):
    for e,file in enumerate(tqdm(restStateDir)):
        matrix=myReshape(pd.read_csv(path+mainDir[1]+'/'+file,header=None).to_numpy())[np.newaxis, :]
        if e == 0:
            restStateAll=matrix 
            continue
        restStateAll=np.concatenate((restStateAll,matrix))
    emptyRoom=pd.read_csv(path+mainDir[0]+'/'+emptyRoomDir[0],header=None).to_numpy()
    emptyRoom = myReshape(emptyRoom) #reshape into [Subjects,PSD,ROI]
    emptyRoomCropped = emptyRoom[:,columns,:]
    restState=np.mean(restStateAll,axis=0)
    restState=restState.take(sort_idx,axis=2)
    psd2use=np.delete(restState,row_idx,axis=0)[:,columns,:] #remove index with nan and select the band-width of interest
    
    # PCA on PSD
    Sub,PSD,ROI=psd2use.shape
    nPCA=10
    restStatePCA=np.zeros((Sub,nPCA,ROI))
    varAcumTot=[]
    
    for roi in range(ROI):
        pca_df, pca2use, prop_varianza_acum= myPCA(np.log(psd2use[:,:,roi]),False,nPCA)
        if roi == 0:
            plt.plot(prop_varianza_acum[:21],'mediumseagreen',alpha=.2, label = 'ROI "n"')
        else:
            plt.plot(prop_varianza_acum[:21],'mediumseagreen',alpha=.2)

        varAcumTot.append(prop_varianza_acum)
        restStatePCA[:,:,roi]=np.array(pca2use)
    plt.plot(np.mean(varAcumTot,0)[:21],'darkslategray',linewidth=3, label = 'Mean explain variance')
    plt.legend()
    plt.vlines(10,0.88, 0.92, 'k', '-')
    plt.hlines(.9,9, 11, 'k','-')
    restStatePCA=RestoreShape(restStatePCA)
    plt.ylabel('Explainded varince ratio')
    plt.xlabel('PCA')
    
    return psd2use, restStatePCA