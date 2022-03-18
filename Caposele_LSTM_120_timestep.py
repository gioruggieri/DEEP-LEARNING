import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_forward=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-look_forward+1):
		a = dataset[i:(i+look_back),0]
		dataX.append(a)
		dataY.append(dataset[i + look_back+look_forward-1, 0])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
dataframe = pd.read_csv('Caposele-Senerchia-Q-P-120.csv', usecols=[1], engine='python', skipfooter=3, delimiter = ';')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1
look_back = 60
look_forward = 60
trainX, trainY = create_dataset(train, look_back, look_forward)
testX, testY = create_dataset(test, look_back, look_forward)
allX, _ = create_dataset(dataset, look_back, 1)
# reshape input to be [samples, time steps, features]
print('grhty', trainX.shape)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
allX = np.reshape(allX, (allX.shape[0], 1, allX.shape[1]))

print(trainX.shape)
print(trainY.shape)
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

all_series_predict = model.predict(allX)
trainY = trainY.reshape(trainY.shape[0], 1)
testY = testY.reshape(testY.shape[0], 1)
print('shape trainy is:', trainY.shape, trainY[0:4])

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
all_series_predict = scaler.inverse_transform(all_series_predict)
# calculate root mean squared error
print(trainY.shape, all_series_predict.shape, dataset.shape)
trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

plt.plot(scaler.inverse_transform(dataset))
all_series_plot = np.empty_like(dataset)
all_series_plot[:, :] = np.nan
all_series_plot = np.vstack((all_series_plot, all_series_plot[0:look_forward,:]))
all_series_plot[look_back+look_forward:len(all_series_predict)+look_back+look_forward, :] = all_series_predict
plt.plot(all_series_plot)
plt.show()
