#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 00:02:20 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model


#%% Gradientes de color =======================================================
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex, n):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

#%% ===========================================================================

def RemoveNan(Data,labels):
    idx=np.argwhere(np.isnan(labels))
    labels=np.delete(labels, idx)
    Data=np.delete(Data, idx,axis=0)
    return Data,labels

#%% ===========================================================================
def myReshape(array):
    [x,y]=array.shape
    newarray=np.zeros((606,300,68))
    for i,j in enumerate(np.arange(0,y,300)):
        newarray[:,:,i]=array[:,j:j+300]
        
    return newarray

#%% ===========================================================================

def RestoreShape(Data):
    if len(Data.shape)>2:
        [x,y,z]=Data.shape
        newarray=np.zeros((x,y*z))
        for i,j in enumerate(np.arange(0,y*z,y)):
            newarray[:,j:j+y]=Data[:,:,i]
        return newarray
    else:
        return Data

#%% ===========================================================================
def myPCA (DataFrame,verbose,nPca):
    from sklearn import preprocessing
    scaled_data = preprocessing.scale(DataFrame)
    pca = PCA() # create a PCA object
    pca.fit(scaled_data) # do the math
    pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
     
    #########################
    #
    # Draw a scree plot and a PCA plot
    #
    #########################
     
    #The following code constructs the Scree plot
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    prop_varianza_acum = pca.explained_variance_ratio_.cumsum()
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
    #the following code makes a fancy looking plot using PC1 and PC2
    pca_df = pd.DataFrame(pca_data, columns=labels)
    pro2use=pca_df.iloc[:,:nPca]
    if verbose:
        
        plt.figure()
        plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        plt.show()
        
        plt.figure()
        plt.scatter(pca_df.PC1, pca_df.PC2)
        plt.title('My PCA Graph')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))
         
        for sample in pca_df.index:
            plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax.plot(
            np.arange(len(labels)) + 1,
            prop_varianza_acum,
            marker = 'o'
        )
    
        for x, y in zip(np.arange(len(labels)) + 1, prop_varianza_acum):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )
            
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(pca.n_components_) + 1)
        ax.set_title('Percentage of cumulative explained variance')
        ax.set_xlabel('PCs')
        ax.set_ylabel('% of cumulative variance');
        plt.show()
        
    return pca_df, pro2use, prop_varianza_acum