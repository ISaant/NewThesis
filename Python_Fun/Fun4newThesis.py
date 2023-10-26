#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 00:02:20 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
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
    cols=y//68
    newarray=np.zeros((x,cols,68))
    for i,j in enumerate(np.arange(0,y,cols)):
        newarray[:,:,i]=array[:,j:j+cols]
        
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

#%% Build graph knn

def knn_graph(connectome, Nneighbours=8):
    
    def build_adjacency(dist, idx):
        """Return the adjacency matrix of a kNN graph."""
        M, k = dist.shape
        assert M,k == idx.shape # comprobaciones para asegurarse de que las dimensiones sean correctas y de que las distancias sean no negativas.
        assert dist.min() >= 0
    
        # Weights.
        sigma2 = np.mean(dist[:, -1])**2 #Calcula el cuadrado de la media de las distancias a los vecinos más cercanos
        # dist = np.exp(- dist**2 / sigma2) # Pasamos de correlacion a distancia, entre mayor la correlacion menor la distancia
    
        # Weight matrix.
        I = np.arange(0, M).repeat(k)
        J = idx.reshape(M*k)
        V = dist.reshape(M*k)
        W = sparse.coo_matrix((V, (I, J)), shape=(M, M)) #Crea una matriz de 
        #coordenadas dispersas (sparse). Es importante para el calculo de los eigenvalues y eigenvectors 
        # No self-connections.
        W.setdiag(0)
        # W= W+W.T
        # Non-directed graph. Asegura que el grafo sea no dirigido al comparar las
        #entradas de la matriz W con sus transpuestas y tomando el mínimo entre 
        #los dos valores. Esto garantiza que las conexiones sean bidireccionales.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)
        return W
    
    _,_,bands=connectome.shape
    for band in range(bands):
        #Making sure that the matrix is actually simetrical
        conn=connectome[:,:,band]
        conn=(conn+conn.T)/2
        #
        idx = np.argsort(-conn)[:, 0:Nneighbours]#se buscan los k valores mas grandes por fila, se busca a partir de la segunda columns porque se le hincharon sus huevos  
        dist = np.array([conn[i, idx[i]] for i in range(conn.shape[0])])
        # dist[dist < 0.1] = 0
        adj_mat_sp = build_adjacency(dist, idx)
        connectome[:,:,band]=adj_mat_sp.toarray()
    
    return connectome
    # fig=plt.figure(figsize=(10,4))
    # fig.add_subplot(121)
    # plt.imshow(adj_mat_sp.todense(), cmap="jet",vmin=0.1, vmax=0.5);
    # plt.colorbar()
    
    # fig.add_subplot(122)
    # plt.hist(adj_mat_sp.data)
    
#%%

def threshold(connectome, tresh):
    tresh=1-tresh
    scaler =MinMaxScaler()
    _,_,bands=connectome.shape
    for band in range(bands):
        X_one_column = connectome[:,:,band].reshape([-1,1])
        result_one_column = scaler.fit_transform(X_one_column)
        bandScaled = result_one_column.reshape(connectome[:,:,band].shape)
        connectome[bandScaled<tresh,band]=0
    return connectome

#%%
def create_Graphs_Disconnected(fcMat):
    _,_,bands=fcMat.shape
    fcDiag=np.zeros((68*bands,68*bands))
    for i in range(bands):
        fcDiag[i*68:68+i*68,i*68:68+i*68]=fcMat[:,:,i]
    
    return fcDiag