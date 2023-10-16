#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:21:50 2023

@author: isaac
"""

''' Autograd: Gradient Computation '''

import torch 
import numpy as np

x= torch.rand(3,requires_grad=True) #the parameter requieres_grad indicates to torch that the values needs to be optimized 
print(x)
y=x+2
print(y)
z=y*y*2
print(z)
v=torch.tensor([.1,1,.001],dtype=torch.float32)
z.backward(v) # dz/dx
print(x.grad) #x.grad is where the gradiants are store (from the jacobian multiplication J . V)

# To modify the tensor so it doesnt need a gradient optimization we can:
#    x.requires_grad(False)
#    y= x.detach()
#    with torch.no_grad():

with torch.no_grad():
    y = x + 2
    print(y)    
    
#%%    
''' DUMMY TRAINING EXAMPLE: Its very important to empty de gradients so they dont accumulate and go to infinity ''' 

weights = torch.ones(4,requires_grad=True)

for i in range(3):
    model_output= (weights*3).sum()
    model_output.backward()
    
    print(weights.grad)    
    weights.grad.zero_()