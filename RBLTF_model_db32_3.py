import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import os
os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#sos.environ['MKL_THREADING_LAYER']='GNU'
# %matplotlib inline

# load the dataset
dataframe = read_csv('apsdiff_train_db32_3.csv', engine='python')
dataset = dataframe.values
# transform int to float
dataset = dataset.astype('float32')
dataset = dataset[:,1:]
# =============================================================================
#print(dataset[0,0],type(dataset))
#print(len(dataset))
# print(len(dataset[0]))
# =============================================================================
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
    return numpy.array(dataX), numpy.array(dataY)

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
            #print(type(min_data),min_data)
    return dataset,min_data,max_data
    #print(type(dataset))
def invert_norm(dataset,min_data,max_data):
    for i in range(len(dataset)):
        dataset[i,0] = (dataset[i,0] * (max_data - min_data)) + min_data
    return dataset
def invert_normtrainy(dataset,min_data,max_data):
    for i in range(len(dataset)):
        dataset[i] = (dataset[i] * (max_data - min_data)) + min_data
    return dataset
look_back=4     
norm_dataset,min_data,max_data = normalize_data(dataset) #数据归一化之后赋给norm_dataset矩阵
train_x,train_y = create_data(norm_dataset,look_back) #将归一化之后的数据分为训练集x和y，两者的区别所选取的行数不同
      
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))#train_x 144x4 个样本，将其变为3D输入，每个时间
trainY = train_y
plt.plot(train_y,'r')
plt.show()                                                         #步长144个样本，我们需要4步，一个特征 
print(type(train_x),len(train_x))
#print(trainX)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=1, verbose=2)
model.save('RBLTFmodel_db32_3_back4.h5')
# make predictions
trainPredict = model.predict(trainX)
trainPredict = invert_norm(trainPredict,min_data,max_data)
trainY = invert_normtrainy(trainY,min_data,max_data)
#print(trainY)
#print(trainPredict[:,0])

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
trainScore1 = math.sqrt(mean_absolute_error(trainY, trainPredict[:,0]))
print('Train Score1: %.2f MAE' % (trainScore1))


plt.plot(trainY,'k')
plt.plot(trainPredict[:,0],'b')
plt.show()

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# convert an array of values into a dataset matrix
# =============================================================================
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return numpy.array(dataX), numpy.array(dataY)
# 
# # fix random seed for reproducibility
# numpy.random.seed(7)
# 
# 
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# datasetest = scaler.fit_transform(datasetest)
# 
# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# 
# # use this function to prepare the train and test datasets for modeling
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# newtestX, newtestY = create_dataset(datasetest, look_back)
# 
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# newtestX = numpy.reshape(newtestX,(newtestX.shape[0], newtestX.shape[1], 1))
# 
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(look_back, 1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
# 
# 
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# newtestPredict = model.predict(newtestX)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# newtestPredict = scaler.inverse_transform(newtestPredict)
# newtestY = scaler.inverse_transform([newtestY])
# 
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# newtestScore = math.sqrt(mean_squared_error(newtestY[0], newtestPredict[:,0]))
# print('NewTest Score: %.2f RMSE' % (newtestScore))
# 
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# 
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2):len(dataset)+1, :] = testPredict
# 
# # shift newtest predictions for plotting
# newtestPredictPlot = numpy.empty_like(datasetest)
# newtestPredictPlot[:,:] = numpy.nan
# newtestPredictPlot[look_back:len(newtestPredict)+look_back, :] = newtestPredict
# 
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
# 
# plt.plot(scaler.inverse_transform(datasetest))
# plt.plot(newtestPredictPlot)
# plt.show()
# 
# 
# =============================================================================
