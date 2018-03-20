#coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math

dataframe = read_csv('翠拥华庭.csv', engine='python', skip_footer=3)



dataframe.columns = ['id','in_time','out_time']
# =============================================================================
# print(dataframe)
# print (dataframe.columns)
# =============================================================================
# add status 1 or -1
dataframe.loc[:,'D'] = 1
dataframe.loc[:,'E'] = -1

# conform in-time and out-time to the same column, status likely.
result2 = pd.concat([dataframe.loc[:,'in_time'], dataframe.loc[:,'out_time']], ignore_index=True)
result3 = pd.concat([dataframe.loc[:,'D'], dataframe.loc[:,'E']], ignore_index=True)
result1 = pd.concat([result2,result3], axis=1)

# generate time series and assign it a mark '*'
date_generate = pd.date_range(start='7/4/2016',end='10/16/2016', freq='10min')
ts = pd.Series('*', index=date_generate)

# DatetimeIndex to concrete colomn value, then concat
date_array = date_generate.to_pydatetime()
# date_only_array = numpy.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array )
date_series = pd.Series(date_array)
result = pd.concat([result1,date_series])


# conform date values of colomn to index
result.set_index(0, inplace = True)
result.index = pd.DatetimeIndex(result.index)

# date_array1 = result.index.to_pydatetime()
# date_only_array = numpy.vectorize(lambda s: s.strftime('%Y-%m-%d %X'))(date_array1)

result.sort_index(inplace=True)
result_list = result.values.tolist()

#calculate the num of parking
parking_num = []
t=[0]
num=0
for i in range(len(result_list)):
    if math.isnan(float(result_list[i][0])):
        result_list[i][0]=0
        t.append(i)
        for j in range(t[-2],t[-1]):
            num = num + float(result_list[j][0])
        parking_num.append(num)

ts = pd.Series(parking_num, index=date_generate)
ts.to_csv('cuiyong_10min.csv', encoding='utf-8', index=True)

#available parking space 
data1 = read_csv('cuiyong_10min.csv',usecols=[1],engine='python').values
# =============================================================================
# print(data1[2],len(data1))
# =============================================================================
AA = []
t= len(data1)
for i in range(t):
    AA.append(int(493-data1[i]))
    #print(type(AA))
pd.DataFrame(AA).to_csv('cyhtaps_10min.csv')

dataframe = read_csv('cyhtaps_10min.csv',engine='python')
a = np.zeros((10,144))
for i in range(10):
    for j in range(144):
        a[i,j] = dataframe.loc[144*7*i+j+144*0,'0']
        
for i in range (10):
    plt.plot(a[i])
plt.show()
# =============================================================================
# print(a)
# b=['00:00','02:00','04:00','06:00','08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00','24:00']
# plt.plot(a[0],label='7.1')
# plt.plot(a[1],label='7.2')
# plt.plot(a[2],label='7.3')
# plt.plot(a[3],label='7.4')
# plt.plot(a[4],label='7.5')
# plt.plot(a[5],label='7.6')
# plt.xticks(np.arange(0,156,12),b,rotation = 45 ,fontsize=10 )
# plt.ylabel('APS',fontsize=12)
# plt.legend()
# plt.grid()
# plt.show()
# =============================================================================



