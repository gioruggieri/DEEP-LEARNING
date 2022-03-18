# LSTM for Q CaposeleRid with regression framing
import numpy
import matplotlib.pyplot as plt
import pandas
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
#from keras.utils.visualize_util import plot

from sklearn.preprocessing import MinMaxScaler

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
	for i in range(len(dataset) - (2*look_back - 1)):
		b = dataset[(i + look_back):(i+2*look_back), 0]
		dataY.append(b)
	return numpy.array(dataX), numpy.array(dataY)

#diadMean calcola la media delle diagonali opposte di una matrice e li inserisce in una lista
def diagMean(x):
	matrix = []
	flipped = numpy.fliplr(x)
	for i in range(-(x.shape[0])+1, x.shape[1]):
		matrix.append(flipped.diagonal(i).mean())
	matrix.reverse()
	return matrix

dataframe = pandas.read_csv('Caposele2.csv', usecols=[0,1], engine='python', skipfooter=0, delimiter = ';')
dataset = dataframe.values
a = dataset.copy()
c = numpy.zeros((int(a[-1,0])-int(a[0,0])+1,2))
x = 0
#ciclo per calcolare i dati mancanti
for i in range(len(a)-1):
	diffx = a[i+1,0] - a[i,0]
	diffy = a[i+1,1] - a[i,1]
	step = float(diffy)/float(diffx)
	c[x,0] = a[i,0]
	c[x,1] = a[i,1]
	x += 1
	if diffx == 1:
		continue

	else:
		for indx in range(int(diffx-1)):
			c[x,0] = c[x-1, 0] + 1
			c[x, 1] = c[x-1, 1] + step
			x+=1
c[-1,0] = a[-1,0]
c[-1,1] = a[-1,1]

data = c[:,1]
data = data.reshape(len(data),1)
print(data.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
datascal = scaler.fit_transform(data)
print('datascal shape is: ', datascal.shape)
train_size = int(len(datascal) * 0.67)
test_size = len(datascal) - train_size
train, test = datascal[0:train_size,:], datascal[train_size:len(datascal),:]
print('train shape is:', train.shape, 'test shape is: ', test.shape)

# reshape into X=t and Y=t+1
look_back = 120
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
#print(trainY[-5:])
#print('________________________')
#print(trainX[-5:])
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
trainX = trainX[:len(trainY),:]
trainY = numpy.reshape(trainY, (trainY.shape[0], trainY.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1]))
testY = numpy.reshape(testY, (testY.shape[0], testY.shape[1]))
testX = testX[:len(testY),:]

print(trainX.shape, trainY.shape, testX.shape, testY.shape, trainX.dtype, trainY.dtype)

# create and fit the Dense network
model = Sequential()
model.add(Dense(2 * look_back, input_dim=(look_back)))
model.add(Dense(look_back))
model.add(Dense(look_back))

#model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
#plot(model, to_file='model.png')
model.fit(trainX, trainY, epochs=1, batch_size=1, shuffle=True, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
trainScore = math.sqrt(trainScore)
trainScore = scaler.inverse_transform(numpy.array([[trainScore]]))
print('Train Score: %.2f RMSE' % (trainScore))

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print(trainPredict.shape)
print(testPredict.shape)

trainPredictPlot = numpy.empty_like(datascal)
trainPredictPlot[:, :] = numpy.nan
trainPredict = diagMean(trainPredict)
trainPredict = numpy.reshape(trainPredict, (len(trainPredict), 1))
testPredict = diagMean(testPredict)
testPredict = numpy.reshape(testPredict, (len(testPredict), 1))
print(trainPredict.shape)
print(testPredict.shape)

trainPredictPlot[look_back+1:len(trainPredict)+look_back+1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(datascal)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(data)-len(testPredict):len(data), :] = testPredict
# plot baseline and predictions
plt.plot(datascal)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



