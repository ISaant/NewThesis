#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:59:04 2023

@author: isaac
"""
import numpy as np


def kill_deadNodes(connectomes):
    assert len(connectomes) == 6
    band_names = list(connectomes.keys())
    connectomes_mod = dict.fromkeys(band_names, [])
    for i, band in enumerate(connectomes):
        band_mod = []
        list_idx = []
        for sub in connectomes[band]:

            idx = np.where(~sub.any(axis=1))[0]
            sub = np.delete(np.delete(sub, idx, axis=0), idx, axis=1)
            band_mod.append(sub)
            list_idx.append(idx)
        connectomes_mod[band_names[i]] = [band_mod, list_idx]

    return connectomes_mod
