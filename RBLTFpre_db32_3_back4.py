# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:52:59 2018

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


df1 = read_csv('cyhtaps_10min.csv',engine='python')
parking_value = np.zeros((5,144))
for i in range(5):
    for j in range(144):
        parking_value[i,j] = df1.loc[144*7*(i+5)+j+144*4,'0']
#print(parking_value)

dataframe = read_csv('apslow_db32_3.csv',engine='python')
#print(dataframe)
a = np.zeros((9,144))
for i in range(9):
    for j in range(144):
        a[i,j] = dataframe.loc[144*7*i+j+144*4,'0']
#print(a)
# =============================================================================
# for x in range(10):
#     plt.plot(a[x]
# plt.show()
# =============================================================================

a_mean = np.zeros((1,144))
for i in range(144):
    a_mean[0,i] = np.mean(a[:,i])
#print(a_mean)

apsdiff_db32_3_pre = np.zeros((5,144))
for i in range(5):
    for j in range(144):
        apsdiff_db32_3_pre[i,j] = parking_value[i,j] - a_mean[0,j]
#print(parking_diff_low)
pd.DataFrame(apsdiff_db32_3_pre).to_csv("apsdiff_db32_3_pre.csv")

# =============================================================================
dataframe = read_csv('apsdiff_db32_3_pre.csv', engine='python')
dataset = dataframe.values
# transform int to float
dataset = dataset.astype('float32')
dataset = dataset[:,1:]
#rint(dataset)
#print(len(dataset))
#print(len(dataset[0]))
# =============================================================================
# plt.plot(dataset)
# plt.show()
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
#print(train_x)
#print(trainX)

model = load_model('RBLTFmodel_db32_3_back4.h5')
trainPredict = model.predict(trainX)
trainPredict = invert_norm(trainPredict,min_data,max_data)
trainY = invert_normtrainy(trainY,min_data,max_data)
#print(trainY)
#print(trainPredict[:,0])

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
trainScore1 = math.sqrt(mean_absolute_error(trainY, trainPredict[:,0]))
print('Train Score1: %.2f MAE' % (trainScore1))

plt.plot(trainY)
plt.plot(trainPredict[:,0])
plt.show()
 
RBLTFaps_text_db32_3 = np.zeros((144,1))
for i in range(144):
        RBLTFaps_text_db32_3[i,0] = trainPredict[i,0]+a_mean[0,i]
#print(RBLTFaps_text_db32_3)

RBLTFaps_true_db32_3 = np.zeros((144,1))
for j in range(144):
        RBLTFaps_true_db32_3[j,0] = df1.loc[144*7*9+j+144*4,'0']
        
np.savetxt("RBLTFaps_text_db32_3.txt",RBLTFaps_text_db32_3)
np.savetxt("RBLTFaps_true_db32_3.txt",RBLTFaps_true_db32_3)


# =============================================================================
# longLSTMaps_pre = np.loadtxt("longLSTMaps_pre.txt")
#  plt.plot(longLSTMaps_pre,'b',label='lstm')
# =============================================================================
# =============================================================================
#  plt.plot(RBLTFaps_true_db32_3,'k',label='real data')
#  plt.plot(RBLTFaps_text_db32_3,'r',label='RBLTF')
#  plt.show()
# =============================================================================
# =============================================================================
#plt.plot(jiauqan,'g')
# =============================================================================
# plt.legend()
# plt.savefig(u'E:/中科院/实验/datachuli/图片/test1.png',dpi=600)
# =============================================================================
# =============================================================================
# plt.show()
# 
# =============================================================================







