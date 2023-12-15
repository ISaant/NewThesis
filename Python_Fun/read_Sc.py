#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:17:01 2023

@author: isaac
"""

import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import netneurotools.networks as netneuro
from tqdm import tqdm
from Fun4newThesis import *
import math
import os
from scipy.stats import pearsonr
def read_Sc(ScFile,path,subjects):

    connectomes = []
    Length = []

    cont = 0
    for file in tqdm(ScFile):
        if len(np.argwhere(subjects == file[:-7])) == 0:
            continue
        mat = scipy.io.loadmat(path+'/'+file)
        # s200_r2_count = mat['schaefer200_radius2_count_connectivity'][:200,:200]
        Length.append(mat['schaefer200_radius2_meanlength_connectivity'][:200, :200])
        connectome = eraseDiag(mat['schaefer200_sift_invnodevol_radius2_count_connectivity'][:200, :200])
        connectomes.append(connectome)

        if cont == 0:
            ROI_names = mat['schaefer200_region_labels'][:200]
        # s200_sift_r2_count = mat['schaefer200_sift_radius2_count_connectivity'][:200,:200]

    hemmid = np.zeros((len(ROI_names), 1))
    for r, ROI in enumerate(ROI_names):
        if ROI[0] == 'L':
            hemmid[r] = 1


    connectomes = np.swapaxes(np.array(connectomes),0,2)
    meanLength = np.mean(Length, axis=0)
    mask_bin = netneuro.struct_consensus(connectomes, meanLength, hemmid, weighted=False) # Aquí eliminamos las conecciones que menos se presentan entre todos los sujetos
    # mask_wei = eraseDiasg(netneuro.struct_consensus(connectomes, meanLength, hemmid, weighted=True))

    connectomes = np.swapaxes(np.array(connectomes),0,2)
    connectomes = np.array([sub * mask_bin for sub in connectomes])


    Corr_wei = []
    plt.figure()
    for adj in tqdm(connectomes):
        corr_wei = []
        steps=np.arange(.01, 1, .01)
        for per in steps:
            mask = percentage(adj,per).squeeze()
            _, corr, _ = measure_sparcity(adj,mask)
            corr_wei.append(corr[0][1])
        corr_wei = np.array(corr_wei)
        Corr_wei.append(corr_wei)

    lower_Tresh=0
    higher_Tresh = 0
    for curve in Corr_wei:
        dist = [math.dist([0, 1], [x, y]) for x, y in zip(steps, curve)]
        idx_min = np.argmin(dist)
        stability_idx = find_positions(curve)
        if idx_min > lower_Tresh: lower_Tresh = idx_min
        if stability_idx > higher_Tresh: higher_Tresh = stability_idx
        plt.scatter(steps[idx_min],curve[idx_min], marker='x', color='olive')
        plt.scatter(steps[stability_idx], curve[stability_idx], marker='x', color='olive')
        plt.plot(steps, curve, color='darkorchid', alpha=.1)
    meanCorr_wei=np.mean(Corr_wei,axis=0)
    dist = [math.dist([0,1],[x,y]) for x,y in zip(steps,meanCorr_wei)]
    idx_min = np.argmin(dist)
    plt.scatter(steps[idx_min], meanCorr_wei[idx_min], marker='x', color='k', s=100 )
    plt.scatter(steps[higher_Tresh], meanCorr_wei[higher_Tresh], marker='x', color='k', s=100)
    plt.plot(steps, meanCorr_wei, color='k', linewidth=4, label='Correlation Thresh vs NoThresh')
    plt.vlines(steps[lower_Tresh], ymin=0, ymax=1.1)
    plt.vlines(steps[higher_Tresh], ymin=0, ymax=1.1)
    plt.show()
    print(lower_Tresh,higher_Tresh)
    for c, connectome in enumerate(connectomes):
        mask = netneuro.threshold_network(connectome,int(steps[higher_Tresh]*100))
        connectome = np.multiply(mask, connectome)
        connectomes[c] = connectome
    return connectomes, Length

def measure_sparcity(data,data_th):
    vec = data.flatten()
    vec_th = data_th.flatten()
    zeros = np.argwhere(vec_th == 0)
    sparcity_bin = len(zeros) / len(vec_th)
    conDen = len(np.argwhere(vec_th > 0))/(len(data)*(len(data)-1))
    # corr = pearsonr(vec, vec_th)
    corr = np.corrcoef(vec,vec_th)

    return sparcity_bin, corr, conDen

def find_positions(array):
    positions = 0

    for i in range(1, len(array)):
        if abs(array[i] - array[i-1]) <= 0.001:
            positions = i
            break

    return positions

def rename(ScFile,path):
    # you can compare the file name with the subjects id inside mat['command']
    for file in tqdm(ScFile):
        subject = file[4:12]
        os.rename(path+'/'+file, path+'/sub_'+subject+'_sc.mat')

