#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:08:45 2023

@author: isaac
"""

#GENERAL TRAING PIPELINE IS DEVIDED AS:
#   1) Design model(input size, output size, forward pass)
#   2) Construct loss and optimizer
#   3) Training loop
#        - forward pass: compute the prediction
#        - backward pass: gradients
#        - update wights

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#  0) prepare data
x_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(y.shape[0],1)

n_samples,n_features=x.shape



# 1)model
input_size=n_features
output_size=1

model = nn.Linear(input_size,output_size)
    
# 2)loss and optimizer

lr=0.01
criterion= nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# training loop

iterations = 300

for epochs in range(iterations):
    
    y_pred= model.forward(x)
    loss=criterion(y_pred,y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epochs %10 == 0:
        print(f'epoch={epochs+1}: loss={loss:.8f}')


predicted= model(x).detach().numpy()
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
# print (f'predictions before training: f(5) = {model(x_test).item():.3f}')
    