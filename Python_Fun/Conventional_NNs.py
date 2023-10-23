#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:03:46 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import seaborn as sns 
from tqdm import tqdm
from FunClassifiers4newThesis_pytorch import NeuralNet,weight_reset,plotPredictionsReg,CustomModel_NoFc
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

def NNs (iterations,num_epochs,dataset,train_size,test_size, batch_size,
              input_size_psd, input_size_anat, output_size,device, lr): 
    
    model = []
    #%%
    NNPred_list_anat=[]
    for i in tqdm(range (iterations)):
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,
                                shuffle=True,num_workers=2)
    
        test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,
                                shuffle=False,num_workers=2)
        if 'model' in globals():
            model.apply(weight_reset)
        model = NeuralNet(input_size_anat, output_size ).to(device)
        criterion = nn.MSELoss()  # Mean Squared Error loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Puedes imprimir el resumen del modelo si lo deseas
        # print(model)
        
        #training loop
        
        n_total_steps = len(train_loader) # number of batches
        for epoch in range (num_epochs):
            for i, (psd, anat, de, th,al, be, g1, g2, target) in enumerate(train_loader):
                anat=anat.to(device)
                target=target.to(device)
                
                #forward 
                outputs=model(anat)
                loss = criterion(outputs,target)
                
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (epoch+1) % 10==0:
                #     print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
                    
        
        #testing and eval
        pred=[]
        with torch.no_grad():
           
            for psd, anat, de, th,al, be, g1, g2, target in test_loader:
                anat=anat.to(device)
                target=target.to(device)
                outputs=model(anat).to('cpu').numpy()
                
                pred.extend(outputs)
        pred=np.array(pred)
        y_test=test_dataset[:][-1].numpy()
        NNPred=plotPredictionsReg(pred.flatten(),y_test.flatten(),False)
        NNPred_list_anat.append(NNPred)
    
    #%%
    NNPred_list_psd=[]
    for i in tqdm(range (iterations)):
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,
                                shuffle=True,num_workers=2)
    
        test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,
                                shuffle=False,num_workers=2)
        if 'model' in globals():
            model.apply(weight_reset)
        model = NeuralNet(input_size_psd, output_size ).to(device)
        criterion = nn.MSELoss()  # Mean Squared Error loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Puedes imprimir el resumen del modelo si lo deseas
        # print(model)
        
        #training loop
        
        n_total_steps = len(train_loader) # number of batches
        for epoch in range (num_epochs):
            for i, (psd, anat, de, th,al, be, g1, g2, target) in enumerate(train_loader):
                psd=psd.to(device)
                target=target.to(device)
                
                #forward 
                outputs=model(psd)
                loss = criterion(outputs,target)
                
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (epoch+1) % 10==0:
                #     print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
                    
        
        #testing and eval
        pred=[]
        with torch.no_grad():
           
            for psd, anat, de, th,al, be, g1, g2, target in test_loader:
                psd=psd.to(device)
                target=target.to(device)
                outputs=model(psd).to('cpu').numpy()
                
                pred.extend(outputs)
        pred=np.array(pred)
        y_test=test_dataset[:][-1].numpy()
        NNPred=plotPredictionsReg(pred.flatten(),y_test.flatten(),False)
        NNPred_list_psd.append(NNPred)
        
    #%%
    NNPred_list_CustomModel_NoFc=[]
    
    
    for i in tqdm(range (iterations)):
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,
                                shuffle=True,num_workers=2)
    
        test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,
                                shuffle=False,num_workers=2)
        if 'model' in globals():
            model.apply(weight_reset)
        model = CustomModel_NoFc(input_size_psd,input_size_anat,output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Puedes imprimir el resumen del modelo si lo deseas
        # print(model)
        
        #training loop
        
        n_total_steps = len(train_loader) # number of batches
        for epoch in range (num_epochs):
            for i, (psd, anat, de, th,al, be, g1, g2, target) in enumerate(train_loader):
                psd=psd.to(device)
                anat=anat.to(device)
                target=target.to(device)
                
                #forward 
                outputs=model(psd,anat)
                loss = criterion(outputs,target)
                
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (epoch+1) % 10==0:
                #     print(f'epoch {epoch} / {num_epochs}, step={i+1}/{n_total_steps}, loss= {loss.item():.4f}')
                    
        
        #testing and eval
        pred=[]
        with torch.no_grad():
           
            for psd, anat, de, th,al, be, g1, g2, target in test_loader:
                psd=psd.to(device)
                anat=anat.to(device)
                target=target.to(device)
                outputs=model(psd,anat).to('cpu').numpy()
                
                pred.extend(outputs)
        pred=np.array(pred)
        y_test=test_dataset[:][-1].numpy()
        NNPred=plotPredictionsReg(pred.flatten(),y_test.flatten(),False)
        NNPred_list_CustomModel_NoFc.append(NNPred)
        
    return NNPred_list_psd,NNPred_list_anat,NNPred_list_CustomModel_NoFc