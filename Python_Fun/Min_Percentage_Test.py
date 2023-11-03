#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:56:20 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Fc_Features import Fc_Feat
from tqdm import tqdm


#%% Min Percentage test

def min_perThresh_test(FcFile,path2fc):
    thresh_vec=np.zeros(6)
    for per in tqdm(np.arange(.2,1,.02)):
        
        delta, theta, alpha, beta, gamma_low, gamma_high, ROIs = Fc_Feat(FcFile,path2fc,per)
        
        is_zero_delta=np.where(~delta.any(axis=1))[0]
        is_zero_theta=np.where(~theta.any(axis=1))[0]
        is_zero_alpha=np.where(~alpha.any(axis=1))[0]
        is_zero_beta=np.where(~beta.any(axis=1))[0]
        is_zero_gamma_low=np.where(~gamma_low.any(axis=1))[0]
        is_zero_gamma_high=np.where(~gamma_high.any(axis=1))[0]

        if is_zero_delta.size == 0 and thresh_vec[0]==0:
            print('true')
            thresh_vec[0]=per
        if is_zero_theta.size == 0 and thresh_vec[1]==0:
            thresh_vec[1]=per
        if is_zero_alpha.size == 0 and thresh_vec[2]==0:
            thresh_vec[2]=per
        if is_zero_beta.size == 0 and thresh_vec[3]==0:
            thresh_vec[3]=per
        if is_zero_gamma_low.size == 0 and thresh_vec[4]==0:
            thresh_vec[4]=per
        if is_zero_gamma_high.size == 0 and thresh_vec[5]==0:
            thresh_vec[5]=per    

    df_per=pd.DataFrame([],columns=['delta','theta','alpha', 'beta', 'gamma_low', 'gamma_high'])
    df_per.loc[0]=thresh_vec
    sns.barplot(df_per, palette='mako')
    plt.title('Min threshold for each band')
    plt.ylabel('percentage')