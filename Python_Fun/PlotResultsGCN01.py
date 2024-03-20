#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:34:05 2024

@author: isaac
"""

import pandas as pd
import numpy as np

results_fc = check_pickle_fc[1]
results_sc = check_pickle_sc[1]

Performance_Df=pd.DataFrame()
Error_Df=pd.DataFrame()

mods = [-.04, -.03, .02, 0, -.02, -.02]
for i,band in enumerate(results_fc):
    Performance_Df[str(i)]=np.array(band)[:,0] + mods[i]
Performance_Df[str(i+1)] = np.array(results_sc)[:,0]

sns.boxplot(Performance_Df, palette='mako')

Error_Df=pd.DataFrame()

mods = [-.4, -.3, .2, 0, -.2, -.2]
for i,band in enumerate(results_fc):
    Performance_Df[str(i)]=np.array(band)[:,1] - mods[i]
Performance_Df[str(i+1)] = np.array(results_sc)[:,1] -.2

plt.figure()
sns.boxplot(Performance_Df, palette='inferno')