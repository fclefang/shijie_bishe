# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:10:46 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas as pd
import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras.models import load_model


df1 = read_csv('cyhtaps_10min.csv',usecols=[1],engine='python')
df1 = df1.values
df1 = df1.astype('float32')
dataset = np.zeros((5,144))
dataset_true = np.zeros((144,1))
for j in range(144):
        dataset_true[j,0] = df1[144*7*9+j+144*0,0]
# =============================================================================
# plt.plot(dataset_true)
# plt.show()
# =============================================================================
# =============================================================================
# print(dataset_true) 
# =============================================================================

for i in range(5):
    for j in range(144):
        dataset[i,j] = df1[144*7*(i+5)+j+144*0,0]
# =============================================================================
# print(dataset)    
# =============================================================================

def create_data(dataset,look_back=1):
    dataX,dataY = [],[]
    for i in range(144):
        for j in range(len(dataset)-look_back):
            a = dataset[j:(j+look_back), i]
            dataX.append(a)
            dataY.append(dataset[j + look_back, i])
    return np.array(dataX), np.array(dataY)

def normalize_data(dataset):
    min_data = 1000
    max_data = -1000
    for i in range(len(dataset)):
        a = min(dataset[i,:])
        b = max(dataset[i,:])
        if a<min_data:
            min_data = a
        if b>max_data:
            max_data = b
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            dataset[i,j] = (dataset[i,j] - min_data) / (max_data - min_data)
    return dataset,min_data,max_data
def invert_norm(dataset,min_data,max_data):
    for i in range(len(dataset)):
        dataset[i,0] = (dataset[i,0] * (max_data - min_data)) + min_data
    return dataset
def invert_normtrainy(dataset,min_data,max_data):
    for i in range(len(dataset)):
        dataset[i] = (dataset[i] * (max_data - min_data)) + min_data
    return dataset
look_back=4     
norm_dataset,min_data,max_data = normalize_data(dataset)
train_x,train_y = create_data(norm_dataset,look_back)
      
# reshape input to be [samples, time steps, features]
trainX = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
trainY = train_y
# =============================================================================
# print(train_x)
# print(trainX)
# =============================================================================

model = load_model('longlstm_back4.h5')
trainPredictl = model.predict(trainX)
trainPredictl = invert_norm(trainPredictl,min_data,max_data)
trainY = invert_normtrainy(trainY,min_data,max_data)
print(trainY)
print(trainPredictl[:,0])

trainScore = math.sqrt(mean_squared_error(trainY, trainPredictl[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
trainScore1 = math.sqrt(mean_absolute_error(trainY, trainPredictl[:,0]))
print('Train Score: %.2f MAE' % (trainScore1))

# =============================================================================
# parking_test_LSTMpre = np.zeros((3,144))
# for i in range(144):
#     for j in range(3):
#         parking_test_LSTMpre [j,i] = trainPredictl[3*i+j,0]
# #print(parking_test_LSTMpre)
# parking_pre_LSTMlong = []
# for i in range(3):
#     for j in range(144):
#         parking_pre_LSTMlong.append(parking_test_LSTMpre[i,:][j])
# 
# parking_true_long = []
# for i in range(3):
#     for j in range(144):
#         parking_true_long.append(dataset_true[i,:][j])
# =============================================================================
        

#print(parking_pre_LSTMlong)
plt.plot(dataset_true)
plt.plot(trainPredictl)
plt.show()

np.savetxt("longLSTMaps_pre.txt",trainPredictl)
