# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:29:14 2018

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

from keras.models import load_model

longLSTMaps_pre = np.loadtxt("longLSTMaps_pre.txt")
RBLTFaps_text_db32_3 = np.loadtxt("RBLTFaps_text_db32_3.txt")
RBLTFaps_true_db32_3 = np.loadtxt("RBLTFaps_true_db32_3.txt")
# =============================================================================
# fig = plt.gcf()
# fig.set_size_inches(10,5)
# b=['8.22','8.29','9.5']
# =============================================================================
plt.plot(RBLTFaps_true_db32_3,'k',label='real data')
plt.plot(longLSTMaps_pre,'b',label='LSTM')
plt.plot(RBLTFaps_text_db32_3,'r',label='RBLTF')

# =============================================================================
# plt.xticks(np.arange(90,432,144),b,rotation = 0 ,fontsize=12 )
# =============================================================================
# =============================================================================
# plt.ylabel('APS',fontsize=12)
# plt.legend()
# plt.grid()
# 
# plt.show()
# =============================================================================

# =============================================================================
# print(a)
# =============================================================================
# =============================================================================
# plt.plot(a[0],label='7.1')
# plt.plot(a[1],label='7.2')
# plt.plot(a[2],label='7.3')
# plt.plot(a[3],label='7.4')
# plt.plot(a[4],label='7.5')
# plt.plot(a[5],label='7.6')
# =============================================================================
b=['00:00','02:00','04:00','06:00','08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00','24:00']
plt.xticks(np.arange(0,156,12),b,rotation = 45 ,fontsize=15 )
plt.ylabel('APS',fontsize=20)
plt.legend()
plt.grid()
fig=plt.gcf()
fig.set_size_inches(8,6)
fig.savefig('pic.jpg',dpi=200)
plt.show()
