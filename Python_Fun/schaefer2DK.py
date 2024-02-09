#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:50:19 2024

@author: isaac
"""

import os
import numpy as np
import pandas as pd 
import scipy
from Fun4newThesis import eraseDiag
def renameLabels_bst(labels):
    renamed_labels=[]
    for i,label in enumerate(labels):
        label = str(labels[i][0])
        renamed_labels.append(label[-1]+'H_'+label[:-2])
    return renamed_labels

def renameLabels_zhen(labels):
    renamed_labels=[]
    for i,label in enumerate(labels):
        label=str(label).split(' ',1)[0]
        renamed_labels.append(label)
    return renamed_labels[:400]

current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path, '../../'))
path2sc = parentPath+'/NewThesis_db_s200/msmtconnectome'
ScFile = np.sort(os.listdir(path2sc))


schaefer=scipy.io.loadmat('Shaefer_400_7net_ROI2Vert.mat')['Shaefer_400_7net_ROI2Vert']
sVertices=np.ndarray.flatten(schaefer['Vertices'])
sLabels_bst=renameLabels_bst(np.ndarray.flatten(schaefer['Label'])) # these are the default labels from brainstorm


DK=scipy.io.loadmat('DK_ROI2Vert.mat')['DK_ROI2Vert']
dkVertices=np.ndarray.flatten(DK['Vertices'])
dkLables=[str(ROI[0]) for ROI in np.ndarray.flatten(DK['Label'])]


file = ScFile[0]
sLabels_Zhen = scipy.io.loadmat(path2sc+'/'+file)['schaefer400_region_labels']
sLabels_Zhen = [str(roi).split(' ',1)[0] for roi in sLabels_Zhen[:400]]

# relationaldf=pd.DataFrame()
# relationaldf['s400_ConnVal']=np.zeros(15000)

reconstructedDK = np.zeros((68,68),float)
intermediateMatrix = np.zeros((15002,15002))

mat = scipy.io.loadmat(path2sc+'/'+file)
connectome = mat['schaefer400_sift_invnodevol_radius2_count_connectivity'][:400, :400]
np.fill_diagonal(connectome,0)
for i,roi in enumerate(sLabels_Zhen):
    roi_idx=np.where(np.array(sLabels_bst)==sLabels_Zhen[i])[0][0]
    vertices = sVertices[roi_idx][0].astype(int)
    
    
        

# connectomes = []
# for file in ScFile:
#     relationalArray=np.zeros(15003)
#     mat = scipy.io.loadmat(path2sc+'/'+file)
#     connectome = mat['schaefer400_sift_invnodevol_radius2_count_connectivity'][:400, :400]
#     np.fill_diagonal(connectome,0)
#     connectomes.append(connectome)   
#     for i,rows in enumerate(connectome):
#         for values in rows:
#             relationalArray[sVertices[i]] = values 
    
#     for i in range(len(reconstructedDK)):
#         for j,vertices in enumerate(dkVertices): 
#             reconstructedDK[i,j] = np.mean(relationalArray[vertices]) 
    
