#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:32:41 2023

@author: isaac
"""
''' GRADIENT DESCENT USING NUMPY '''

import torch
import torch.nn as nn 
import numpy as np

''' for this we will use the linear regression example again'''

# f = w * x

x = np. array ([1,2,3,4], dtype=np.float32)
y = np. array ([2,4,6,8], dtype=np.float32)

w= 0.0

#model predictions
def forward(x):
    return x*w

#loss = MSE
def loss (y,y_predicted):
    return ((y_predicted-y)**2).mean()
     
#gradient 
# MSE = 1/N * (w*x)**2
#dJ/dw=1/N(2x(w*x-y)) <--- chain rule

def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

#print predictions before training
print (f'predictions before training: f(5) = {forward(5):.3f}')
    
#training
lr= 0.01
n_iters=20

for epoch in range(n_iters):
    #forward
    y_pred=forward(x)
    
    #loss
    l=loss(y,y_pred)
    
    #gradients
    dw=gradient(x, y, y_pred)
    
    #update formula 
    w-=lr*dw
    
    if epoch %1 == 0:
        print(f'epoch={epoch+1}: w={w:.3f}, loss={l:.8f}')
        

print (f'predictions after training: f(5) = {forward(5):.3f}')

#%% 

''' GRADIENT DESCENT USING AUTOGRAD '''

x = torch.tensor ([1,2,3,4], dtype=torch.float32)
y = torch.tensor ([2,4,6,8], dtype=torch.float32)

w= torch.tensor(0.0,dtype=torch.float32, requires_grad=True)

#model predictions
def forward(x):
    return x*w

#loss = MSE
def loss (y,y_predicted):
    return ((y_predicted-y)**2).mean()


print (f'predictions before training: f(5) = {forward(5):.3f}')
    
#training
lr= 0.01
n_iters=70 #we need more iterations because the gradient is not as exact as the numerical approach

for epoch in range(n_iters):
    #forward
    y_pred=forward(x)
    
    #loss
    l=loss(y,y_pred)
    
    #gradients
    l.backward() #dl/dw
    
    #update formula 
    with torch.no_grad():
        w-=lr*w.grad 
        
    #empty gradients
    w.grad.zero_()
    
    
    if epoch %5 == 0:
        print(f'epoch={epoch+1}: w={w:.3f}, loss={l:.8f}')
        
print (f'predictions before training: f(5) = {forward(5):.3f}')

#%% 
#GENERAL TRAING PIPELINE IS DEVIDEN AS:
#   1) Design model(input size, output size, forward pass)
#   2) Construct loss and optimizer
#   3) Training loop
#        - forward pass: compute the prediction
#        - backward pass: gradients
#        - update wights
x = torch.tensor ([[1],[2],[3],[4]], dtype=torch.float32) # num of rows in the num of samples 
y = torch.tensor ([[2],[4],[6],[8]], dtype=torch.float32)

x_test=torch.tensor([5],dtype=torch.float32)
n_samples,n_features=x.shape
print(x.shape)

input_size=n_features
output_size=n_features
# model = nn.Linear(input_size,input_size)

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.lin = nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        return self.lin(x)

model=LinearRegression(input_size,output_size)
loss= nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr)
print (f'predictions before training: f(5) = {model(x_test).item():.3f}')
    
#training
lr= 0.01
n_iters=250 #we need more iterations because the gradient is not as exact as the numerical approach

for epoch in range(n_iters):
    #forward
    y_pred=model.forward(x)
    
    #loss
    l=loss(y,y_pred)
    
    #gradients
    l.backward() #dl/dw
    
    #update formula 
    optimizer.step() 
        
    #empty gradients
    optimizer.zero_grad()
    
    
    if epoch %10 == 0:
        [w,b]=model.parameters()
        print(f'epoch={epoch+1}: w={w[0][0].item():.3f}, loss={l:.8f}')
        
print (f'predictions before training: f(5) = {model(x_test).item():.3f}')

