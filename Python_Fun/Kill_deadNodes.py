#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:59:04 2023

@author: isaac
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
def kill_deadNodes(connectome):
    
    list_connectomes_mod=[]
    list_idx=[]
    for sub in connectome:

        idx=np.where(~sub.any(axis=1))[0]
        sub=np.delete(np.delete(sub, idx, axis=0) , idx, axis=1)
        list_connectomes_mod.append(sub)
        list_idx.append(idx)
        
    return list_connectomes_mod, list_idx