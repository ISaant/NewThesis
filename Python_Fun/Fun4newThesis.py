#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 00:02:20 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from copy import copy
import seaborn as sns
sns.set_context("talk")
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model, svm, preprocessing



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
def myReshape(array,rois=68):
    [x,y]=array.shape
    cols=y//rois
    newarray=np.zeros((x,cols,rois))
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
def PltDistDemographics(demographics):
    sns.violinplot(data=demographics, x='Sexo', y='Edad', palette='mako', inner="point")
    plt.title('Distribución de las edades por genero')
    Age = demographics['Edad'].to_numpy()
    Acer = demographics['Acer'].to_numpy()
    Cattell = demographics['Cattell'].to_numpy()
    # RoundAge = copy(Age)
    # RoundAge[RoundAge < 30] = 30
    # for i in np.arange(30, 90, 10):
    #     print(i)
    #     RoundAge[np.logical_and(RoundAge > i, RoundAge <= i + 10)] = (i + 10)
    # # RoundAge[RoundAge>80]=90
    # demographics['Intervalo'] = RoundAge
    sns.displot(data=demographics, x='Cattell', hue='Intervalo', kind='kde', fill=True)
    plt.title('Distribución de Cattell por rango de edad')
    plt.ylabel('Densidad')
    sns.displot(data=demographics, x='Acer', hue='Intervalo', kind='kde', fill=True)
    plt.title('Distribución de ACE-R por rango de edad')
    plt.ylabel('Densidad')
    plt.xlim([60, 110])
    # plt.figure()
    # sns.lmplot(x='Age', y='Cattell', data=demographics,
    #            scatter=False, scatter_kws={'alpha': 0.3}, palette='CMRmap')
    plt.figure()
    sns.residplot(data=demographics, x="Edad", y="Cattell", order=2, line_kws=dict(color="r"))
    # plt.figure()
    # sns.residplot(data=demographics, x="Edad", y="Cattell", order=2, line_kws=dict(color="r"))
    sns.relplot(data=demographics, y='Cattell', x='Edad', hue='Intervalo')
    plt.title('Regresión de Cattell con respecto a la Edad ')
    rsq, pvalue = scipy.stats.pearsonr(Age, Cattell)
    rsq_cuad, pvalue_cuad = scipy.stats.pearsonr(Age**2, Cattell**2)
    Age = Age.reshape(-1, 1)
    linReg = linear_model.LinearRegression()
    # linReg.fit(Edad, Cattell)
    # Predict data of estimated models
    # line_age = np.round(np.arange(Age.min() - 5, Age.max() + 5, .01), 2)[:, np.newaxis]
    # line_predCatell = linReg.predict(line_age)

    regressor = svm.SVR(kernel='poly',degree=2)
    regressor.fit(Age, Cattell)
    curve_age = np.round(np.arange(Age.min()-5, Age.max() +5, .01), 2)[:, np.newaxis]
    curve_predCatell = regressor.predict(curve_age)
    plt.plot(curve_age, curve_predCatell, color="olive", linewidth=4, alpha=.7)


    plt.annotate('PearsonR= ' + str(round(rsq_cuad, 2)),
                 (20, 15), fontsize=12)
    # plt.annotate('pvalue= ' + str(round(pvalue_cuad,4)),
    #              (20, 12), fontsize=12)
    plt.annotate('pvalue < .0001',
                 (20, 12), fontsize=12)

    Residuals = returnResuduals(demographics, ['Cattell'], linReg)
    Residuals ['Intervalo'] = demographics['Intervalo']
    # dfRes_melt = pd.melt(Residuals, id_vars=['Edad'],
    #                      value_vars=['resCattell', 'Intervalo'])

    color = 'mako'

    sns.displot(data=Residuals, x='resCattell', hue='Intervalo', kind='kde',
                fill=True, palette=color)

    sns.lmplot(x='Edad', y='resCattell', data=Residuals,
               scatter=False, scatter_kws={'alpha': 0.3}, palette=color)

    sns.relplot(data=demographics, x='Cattell', y='Acer', hue='Intervalo')
    plt.title('Cattell-Acer Regression')
    rsq, pvalue = scipy.stats.pearsonr(Cattell, Acer)
    Cattell = Cattell.reshape(-1, 1)
    # Acer=Acer.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Cattell, Acer)
    # Predict data of estimated models
    line_X = np.linspace(Cattell.min(), Cattell.max(), 603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    plt.plot(line_X, line_y, color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR= ' + str(round(rsq, 2)),
                 (20, 17), fontsize=12)
    plt.annotate('pvalue= ' + str(round(pvalue,4)),
                 (20, 10), fontsize=12)


    sns.relplot(data=demographics, x='Edad', y='Acer', hue='Intervalo')
    plt.title('Regresión de ACE-R con respecto a la Edad')

    regressor.fit(Age, Acer)
    curve_predAcer = regressor.predict(curve_age)
    plt.plot(curve_age, curve_predAcer, color="olive", linewidth=4, alpha=.7)
    plt.ylabel('ACE-R')
    Age = Age.reshape(Age.shape[0], )
    rsq, pvalue = scipy.stats.pearsonr(Age**2, Acer**2)
    plt.annotate('PearsonR= ' + str(round(rsq, 2)),
                 (20, 77), fontsize=12)
    # plt.annotate('pvalue= ' + str(round(pvalue_cuad, 4)),
    #              (20, 70), fontsize=12)
    plt.annotate('pvalue < .0001 ',
                 (20, 70), fontsize=12)

    Age = Age.reshape(Age.shape[0], )
    rsq, pvalue = scipy.stats.pearsonr(Age, Acer)
    Age = Age.reshape(-1, 1)
    # linReg = linear_model.LinearRegression()
    # linReg.fit(Age, Acer)
    # Predict data of estimated models
    # line_X = np.linspace(Age.min(), Age.max(), 603)[:, np.newaxis]
    # line_y = linReg.predict(line_X)
    # plt.plot(line_X, line_y, color="yellowgreen", linewidth=4, alpha=.7)
    # plt.annotate('PearsonR= ' + str(round(rsq, 2)),
    #              (20, 77), fontsize=12)
    # plt.annotate('pvalue= ' + str(pvalue),
    #              (20, 70), fontsize=12)
    plt.ylim([60, 110])

    plt.show()
    # return line_age, line_predCatell

#%% ==========================================================================
def returnResuduals(df, Variables, model):
    x = np.array(copy(df['Edad'])).reshape(-1, 1)
    resDf = copy(df)

    for var in Variables:
        nanidx = np.array(np.where(np.isnan(df[var]))).flatten()
        y = np.array(df[var].fillna(df[var].mean()))  # fill the nan with the mean value... not sure if its the best solution
        model.fit(x, y)
        # Predict data of estimated models
        predictions = model.predict(x)
        residuals = y - predictions
        resDf[var] = residuals
        resDf.loc[nanidx, var] = np.nan
        resDf.rename(columns={var: 'res' + var}, inplace=True)
    return resDf

#%% ===========================================================================
def myPCA (DataFrame,verbose,nPca):
    from sklearn import preprocessing
    # scaled_data = preprocessing.scale(DataFrame)
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(DataFrame)
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
        dist = np.exp(- dist**2 / sigma2) # Pasamos de correlacion a distancia, entre mayor la correlacion menor la distancia


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
        adj_mat_sp = build_adjacency(dist, idx).toarray()
        connectome[:,:,band]=adj_mat_sp
    
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

def percentage(connectome, per):
    data = copy(connectome)
    if data.ndim < 3:
        data = data[:, :, np.newaxis]
    x,y,bands = data.shape
    for band in range(bands):
        upper_triangle = data[:, :, band]
        upper_triangle = upper_triangle[np.triu_indices(upper_triangle.shape[0], k=1)]
        num_of_edges = np.ceil(len(upper_triangle)*per).astype(int)
        idx = np.argsort(-upper_triangle)[num_of_edges:]
        upper_triangle[idx] = 0
        X = np.zeros((x,y))
        X[np.triu_indices(X.shape[0], k=1)] = upper_triangle
        data[:, :, band] = X + X.T - np.diag(np.diag(X))
    return data

#%%
def create_Graphs_Disconnected(fcMat):
    _,_,bands=fcMat.shape
    fcDiag=np.zeros((68*bands,68*bands))
    for i in range(bands):
        fcDiag[i*68:68+i*68,i*68:68+i*68]=fcMat[:,:,i]
    
    return fcDiag

#%% 
def CorrHist(FcFile,path):
    for e,file in enumerate(tqdm(FcFile)):
        fcMat = scipy.io.loadmat(path+'/'+file)['TF_Expand_Matrix_Sorted']
        scaler =MinMaxScaler()
        x,y,bands=fcMat.shape
        
        vecs= []
        for band in range(bands):
            fcMat_vec = fcMat[:,:,band].reshape([-1,1])
            # fcMat_vec -= min(fcMat_vec)
            # fcMat_vec /= max(fcMat_vec)
            # fcMat_vec = scaler.fit_transform(fcMat_vec)
            vecs.append(fcMat_vec)
        vecs=np.array(vecs)
        if e==0:
            fcMat_Vec=vecs
        else: 
            fcMat_Vec=np.concatenate((fcMat_Vec,vecs),axis=1)
    fcMat_Vec= np.squeeze(fcMat_Vec)       
    print (fcMat_Vec.shape)
            
    fig, ax = plt.subplots(nrows=2, ncols=3,figsize=(15, 15))
    
    count=0
    title=['delta', 'theta', 'alpha', 'beta', 'gamma_low', 'gamma_high']
    for row in ax:
        for col in row:
            col.hist(fcMat_Vec[count,:], bins=30, range= (0.01,max(fcMat_Vec[count,:])),density=True, )
            col.hist(fcMat_Vec[count,:], bins=30, range= (0.01,max(fcMat_Vec[count,:])),density=True, cumulative= True, histtype= 'step')
            col.set_title(title[count])
            count+=1
#%%
def eraseDiag(matrix):
    for node in range(matrix.shape[0]):
        matrix[node,node]=0
    return matrix

        
        
        
        
        
        
        
        
        
        
        
        
        