#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:36:19 2023

@author: isaac
"""
import os
import numpy as np

path='/media/isaac/Elements/camcan_AEC_ortho'

Dir=np.sort(os.listdir(path))
Sub=[x[0:12] for x in Dir]
unique,count=np.unique(Sub,return_counts=True)
where=np.argwhere(np.array(Sub)==unique[46])