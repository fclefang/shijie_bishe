# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:26:22 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd

df1 = read_csv('cyhtaps_10min.csv',engine='python')
parking_value = np.zeros((9,144))
for i in range(9):
    for j in range(144):
        parking_value[i,j] = df1.loc[144*7*i+j+144*0,'0']

for x in range(9):
    plt.plot(parking_value[x])
plt.show()
# =============================================================================
# print(parking_value)
# =============================================================================

dataframe = read_csv('apslow_db32_3.csv',engine='python')
# =============================================================================
# print(dataframe)
# =============================================================================
a = np.zeros((9,144))
for i in range(9):
    for j in range(144):
        a[i,j] = dataframe.loc[144*7*i+j+144*0,'0']
# =============================================================================
# print(a)
# =============================================================================
for x in range(9):
    plt.plot(a[x])
plt.show()
 
a_mean = np.zeros((1,144))
for i in range(144):
    a_mean[0,i] = np.mean(a[:,i])
    
# =============================================================================
# plt.plot(a_mean[0])
# plt.show()
# =============================================================================
# =============================================================================
# print(a_mean)
# =============================================================================

apsdiff_train_db32_3 = np.zeros((9,144))
for i in range(9):
    for j in range(144):
        apsdiff_train_db32_3[i,j] = parking_value[i,j] - a_mean[0,j]
# =============================================================================
# print(apsdiff_train_db32_3)
# =============================================================================
for x in range(9):
    plt.plot(apsdiff_train_db32_3[x])
plt.show()
pd.DataFrame(apsdiff_train_db32_3).to_csv("apsdiff_train_db32_3.csv")

