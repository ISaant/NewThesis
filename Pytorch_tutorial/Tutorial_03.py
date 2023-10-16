#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:28:40 2023

@author: isaac
"""
''' HOW TO COMPUTE GRADIENT USING CHAIN RULE '''
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w=torch.tensor(1.0, requires_grad=True)

y_hat=x*w
s=y_hat-y
loss=s**2

loss.backward()
print(w.grad)

## update weights
## iterate 

