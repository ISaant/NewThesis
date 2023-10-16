#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:30:19 2023

@author: isaac
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#  0) prepare data
bc = datasets.load_breast_cancer()
x,y= bc.data,bc.target

n_samples,n_features=x.shape

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=1234)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)


# 1) model

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward (self,x):
        y_predicted=torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)
    
# 2) loss and optimizer

lr=0.01
criterion= nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3) training loop

iterations = 300

for epochs in range(iterations):
    #forward pass
    y_pred= model.forward(x_train)
    loss=criterion(y_pred,y_train)
    
    #backward  pass
    loss.backward()
    
    #weights updates
    optimizer.step()
    optimizer.zero_grad()
    if epochs %10 == 0:
        print(f'epoch={epochs+1}: loss={loss:.8f}')

with torch.no_grad():
    y_predicted=model(x_test)
    y_predicted_cls=y_predicted.round()
    acc= y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])

print(acc)