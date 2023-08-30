#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 13:56:05 2023

@author: isaac
"""

import tensorflow as tf
import scipy 
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from tensorflow import keras
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
# Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
# Memory growth must be set before GPUs have been initialized
        print(e)

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D,Conv1D, concatenate
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import Model
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#%% ===========================================================================
def Scale(Data):
    
    scaler=StandardScaler()
    scaler.fit(Data)
    Data=scaler.transform(Data)
    return Data

#%% ===========================================================================
def Split(Data,labels,testSize,seed=None):
    idx = np.arange(len(Data))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(Data, labels, idx, test_size=testSize,random_state=seed)
    return  x_train, x_test, y_train, y_test, idx_train, idx_test

#%% ===========================================================================
def Perceptron_PCA (Input0,numOutputNeurons):
    # print(classification)
    tf.keras.backend.clear_session()
    NN0 = Dense(512, activation='sigmoid')(Input0)
    NN0 = Dense(256, activation='relu')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    # NN0 = Dense(1024, activation='relu')(NN0)
    # NN0 = Dense(32, activation='relu')(NN0)
    output = Dense(numOutputNeurons, activation='linear')(NN0)
    loss='mean_squared_error',
    metrics=['mape']
    model = Model(
        inputs=Input0,
        outputs=output)
    
    
    # print(model.summary())
    model.compile(optimizer=Adam(learning_rate=.0001),
                  loss=loss,
                  metrics=metrics)

    return model

#%% ===========================================================================

def Perceptron (Input0,numOutputNeurons):
    # print(classification)
    tf.keras.backend.clear_session()
    NN0 = Dense(512, activation='relu')(Input0)
    NN0 = Dense(256, activation='relu')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    output = Dense(numOutputNeurons, activation='linear')(NN0)
    loss='mean_squared_error',
    metrics=['mape']
    model = Model(
        inputs=Input0,
        outputs=output)
    
    
    # print(model.summary())
    model.compile(optimizer=Adam(learning_rate=.0001),
                  loss=loss,
                  metrics=metrics)

    return model
#%% ===========================================================================

def parallelNN (PSD, Anat,Fc,numOutputNeurons):
    
    InputPSD=tf.keras.Input(shape=(PSD.shape[1],), )
    InputAnat=tf.keras.Input(shape=(Anat.shape[1],), )
    tf.keras.backend.clear_session()
    NN0 = Dense(512, activation='sigmoid')(InputPSD)
    NN0 = Dense(256, activation='relu')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    
    NN1 = Dense(512, activation='relu')(InputAnat)
    NN1 = Dense(256, activation='relu')(NN1)
    NN1 = Dense(64, activation='relu')(NN1)
    NN1 = Dense(16, activation='relu')(NN1)
    
    # NN2 = Dense(512, activation='relu')(InputFc)
    # NN2 = Dense(128, activation='sigmoid')(NN2)
    # NN2 = Dense(128, activation='relu')(NN2)
    # NN2 = Dense(128, activation='tanh')(NN2)
    # NN2 = Dense(64, activation='relu')(NN2)
    # NN2 = Dense(32, activation='relu')(NN2)
    
    x = concatenate([NN0,NN1])
    
    Prob_Dense = Dense(32, activation='relu',name="Last_NN_Targets")(x)
    # Prob_Dense = Dropout(.3)(Prob_Dense)
    Prob_Dense = Dense(16, activation='relu')(Prob_Dense)
    output = Dense(numOutputNeurons, activation='linear',name='output')(Prob_Dense)
    
    model = Model(inputs=[InputPSD,InputAnat],
                outputs=output)
    
    loss='mean_squared_error',
    metrics=['mape']
    
    model.compile(optimizer=Adam(learning_rate=.001),
                  loss=loss,
                  metrics=metrics)
    keras.utils.plot_model(model, "multi_input_and_output_model_1.png", show_shapes=True)
    
    return model

#%% ===========================================================================

def parallelNN2p0 (PSD, Anat,Fc):
    
    InputPSD=tf.keras.Input(shape=(PSD.shape[1],), )
    InputAnat=tf.keras.Input(shape=(Anat.shape[1],), )
    tf.keras.backend.clear_session()
    NN0 = Dense(512, activation='sigmoid')(InputPSD)
    NN0 = Dense(256, activation='relu')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    
    NN1 = Dense(512, activation='relu')(InputAnat)
    NN1 = Dense(256, activation='relu')(NN1)
    NN1 = Dense(64, activation='relu')(NN1)
    NN1 = Dense(16, activation='relu')(NN1)
    
    # NN2 = Dense(512, activation='relu')(InputFc)
    # NN2 = Dense(128, activation='sigmoid')(NN2)
    # NN2 = Dense(128, activation='relu')(NN2)
    # NN2 = Dense(128, activation='tanh')(NN2)
    # NN2 = Dense(64, activation='relu')(NN2)
    # NN2 = Dense(32, activation='relu')(NN2)
    
    x = concatenate([NN0,NN1])
    
    Prob_Dense = Dense(32, activation='relu',name="Last_NN_Targets")(x)
    # Prob_Dense = Dropout(.3)(Prob_Dense)
    Prob_Dense = Dense(16, activation='relu')(Prob_Dense)
   
    O1=Dense(8,activation='relu')(Prob_Dense)
    O1=Dense(1,activation='linear')(O1)
    
    O2=Dense(8,activation='relu')(Prob_Dense)
    O2=Dense(1,activation='linear')(O2)
    
    O3=Dense(8,activation='relu')(Prob_Dense)
    O3=Dense(1,activation='linear')(O3)
    
    O4=Dense(8,activation='relu')(Prob_Dense)
    O4=Dense(1,activation='linear')(O4)
    
    O5=Dense(8,activation='relu')(Prob_Dense)
    O5=Dense(1,activation='linear')(O5)
    
    O6=Dense(8,activation='relu')(Prob_Dense)
    O6=Dense(1,activation='linear')(O6)
    
    O7=Dense(8,activation='relu')(Prob_Dense)
    O7=Dense(1,activation='linear')(O7)
    
    O8=Dense(8,activation='relu')(Prob_Dense)
    O8=Dense(1,activation='linear')(O8)
    
    O9=Dense(8,activation='relu')(Prob_Dense)
    O9=Dense(1,activation='linear')(O9)
    
    
    
    model = Model(inputs=[InputPSD,InputAnat],
                outputs=[O1,O2,O3,O4,O5,O6,O7,O8,O9])
    
    loss='mean_squared_error',
    metrics=['mape']
    
    model.compile(optimizer=Adam(learning_rate=.001),
                  loss=loss,
                  metrics=metrics)
    
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    
    return model
#%% ===========================================================================
def trainModel(model,x_train,y_train,epochs,plot):
    keras.backend.clear_session()
    history = model.fit(x_train, 
                        y_train, 
                        validation_split=0.2, 
                        batch_size=64,
                        epochs =epochs,
                        verbose=0)
    
    if plot:
        
        plt.figure()
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

#%% ===========================================================================
def evaluateRegModel(model,x_test,y_test):
    mse_neural, mape_neural = model.evaluate(x_test, y_test, verbose=0)
    # print('Mean squared error from neural net: ', mse_neural)
    # print('Mean absolute percentage error from neural net: ', mape_neural)
    predictions = model.predict(x_test).flatten()
    return predictions

#%% ===========================================================================
def evaluateClassModel(model,x_test,y_test):
    # print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    print(results)
    
#%% Function to plot predictions

def plotPredictionsReg(predictions,y_test,plot):
    pearson=scipy.stats.pearsonr(predictions,y_test)
    if plot :
        plt.figure()
        plt.scatter(predictions,y_test)
        
        # print(pearson)
        lims=[min(y_test)-1,max(y_test)+1]
        plt.plot(lims,lims)
        plt.xlabel('predicted')
        plt.ylabel('ture values')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()
    return pearson[0]