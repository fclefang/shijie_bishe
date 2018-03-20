import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#os.environ['MKL_THREADING_LAYER']='GNU'
# %matplotlib inline

# load the dataset

df1 = read_csv('cyhtaps_10min.csv',usecols=[1],engine='python')
df1 = df1.values
df1 = df1.astype('float32')
dataset = np.zeros((9,144))
for i in range(9):
    for j in range(144):
        dataset[i,j] = df1[144*7*i+j+144*0,0]
# =============================================================================
# print(dataset)
# =============================================================================


# transform int to float


# =============================================================================
# plt.plot(dataset)
# plt.show()
# 
# =============================================================================

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# convert an array of values into a dataset matrix
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
print(train_x)
print(trainX)

model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)
model.save('longlstm_back4.h5')
# make predictions
trainPredict = model.predict(trainX)
trainPredict = invert_norm(trainPredict,min_data,max_data)
trainY = invert_normtrainy(trainY,min_data,max_data)
print(trainY)
print(trainPredict[:,0])

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))


plt.plot(trainY)
plt.plot(trainPredict[:,0])
plt.show()

# =============================================================================
# model = load_model('mylstm_back4.h5')
# trainPredict = model.predict(trainX)
# trainPredict = invert_norm(trainPredict,min_data,max_data)
# trainY = invert_normtrainy(trainY,min_data,max_data)
# print(trainY)
# print(trainPredict[:,0])
# 
# trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# 
# plt.plot(trainY)
# plt.plot(trainPredict[:,0])
# plt.show()
# =============================================================================
