#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:19:18 2023

@author: isaac
"""

"""
Learn how to create tensors and how to go from numpy to tensor and back

"""

import torch
import numpy as np

x=torch.zeros(3,2,2,dtype=torch.int)
print(x,x.dtype)

x=torch.tensor([1,2,3,4])
print(x.dtype)

x=torch.rand(2,2)
y=torch.rand(2,2)

z= x+y
print(z)

z=y.add_(x) #ALL THE FUNCTIONS WITH A UNDERSCORE WILL BE PERFORMED INPLACE 
print(z)

x=torch.rand(4,4)
y=x.view(16) #Reshape
print(y)
y=x.view(-1,8) #reshape but pytorch will find the best solution for the first dim
print(y)

# new_tensor = tensor.reshape(6, 2) is the same as new_tensor = tensor.view(6, 2)

'''Note'''  # When transforming into numpy, if the tensor is allocated in the cpu
            # the transformed tensor (now array) will be allocated in the same 
            # memory space
            
a= torch.ones(5)
b=a.numpy()

a.add_(1)
print(a,b)

a=np.ones(5)
b=torch.from_numpy(a).to(torch.int)

a+=1
print(a,b)

'''Note'''  #If we want to make an operantion on the gpu and the move it to 
            # a numpy structure, its important to mo it first to the cpu 
            # (after the operation) because numpy can only handle cpu tensors
if torch.cuda.is_available():
    device = torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y
    # z.numpy() # this will arrise an error
    z=z.to('cpu')
    z.numpy()
    z+=1
    print(z)