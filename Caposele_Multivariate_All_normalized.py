import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def numpy_minmax(X, Xmin, Xmax):
	return (X - Xmin) / (Xmax - Xmin)


def numpy_minmax_inv(X, Xmin, Xmax):
	return X*(Xmax-Xmin)+Xmin

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, look_f=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		b = dataset[i:(i + look_back), 1]
		a = np.hstack((a, b))
		dataX.append(a)
	for i in range(len(dataset) - (look_back + look_f - 1)):
		b = dataset[(i + look_back):(i + look_back + look_f), 0]
		dataY.append(b)
	return np.array(dataX), np.array(dataY)

def NSE(y_pred, y, optim = False):

	'''
	Nash-Sutcliffe Efficiency criterion
	Parameters
	-----------
	optim : bool
		if True, the objective is translated to be used in optimizations
		where a minimum value is seeked by the algorithm
	Notes
	--------
	Widely used criterion in hydrology, values ranging from -infty -> 1
	A zero value means the model is not better than the 'no knowledge'
	model, which is characterised by the mean of the observations.
	Sensitive to extreme values.
	* range: [-inf, 1]
	* optimum: 1
	* category: comparison with reference model
	'''
	residuals = y - y_pred
	nom = np.sum(residuals**2)
	den = np.sum((y - np.mean(y))**2)
	OF = 1. - nom/den

	if optim == True:
		OF = 1. - OF
	return OF
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
dataframe = pd.read_csv('Caposele-Senerchia-Q-P.csv', usecols=[1, 2], engine='python', skipfooter=3, delimiter=';')
dataframe1 = dataframe
dataset = dataframe1.values
datasetor = dataset.astype('float32')

dataset_min = dataset[:, 0].min()
dataset_max = dataset[:, 0].max()
dataset1_min = dataset[:, 1].min()
dataset1_max = dataset[:, 1].max()
dataset1 = numpy_minmax(dataset[:, 0], dataset_min, dataset_max)
dataset2 = numpy_minmax(dataset[:, 1], dataset1_min, dataset1_max)
dataset = np.column_stack((dataset1, dataset2))
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, ], dataset[train_size:len(dataset), ]

# reshape into X=t and Y=t+1
look_back = 60
look_f = 60
trainX, trainY = create_dataset(train, look_back, look_f)
testX, testY = create_dataset(test, look_back, look_f)
trainX = trainX[:len(trainY), :]
print(trainX.shape)

# reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2], 1))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], trainX.shape[2], 1))
# trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1]))
# create and fit the LSTM network
print('trainY is:', trainY.shape)
print('datasetor is :', datasetor.shape, 'trainX is: ', trainX.shape, 'testX is: ', testX.shape, 'testY is:',
	  testY.shape, 'traindataset is', dataset[train_size:len(dataset), ].shape)

# testY = np.reshape(testY, (testY.shape[0], testY.shape[1]))
testX = testX[:len(testY), :]

testLoss1 = []
testAcc1 = []


class LossHistory(keras.callbacks.Callback):
	def on_epoch_begin(self, epoch, logs=None):
		self.losses = []

	def on_epoch_end(self, epoch, logs={}):
		testLoss = model.evaluate([testX], [testY], verbose=0)
		print('testLoss is:', testLoss)
		testLoss1.append(testLoss[0])
		testAcc1.append(testLoss[1])
		self.losses.append(logs.get('loss'))


main_input = Input(shape=(2 * look_back,), name='main_input')
dense_out1 = (Dense(1000, activation='sigmoid', name='dense_out1'))(main_input)
dense_out = (Dense(1000, activation='sigmoid', name='dense_out'))(dense_out1)
main_output = Dense(look_f, activation='sigmoid', name='aux_output')(dense_out)
model = Model(inputs=[main_input], outputs=[main_output])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history1 = LossHistory()
history = model.fit([trainX], [trainY], epochs=30, batch_size=32, callbacks=[history1], verbose=1)
#plot_model(model, to_file='model.png')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(testAcc1, color='red')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(testLoss1, color='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = trainPredict[:, look_f-1]
testPredict = testPredict[:, look_f-1]
trainPredict = numpy_minmax_inv(trainPredict, dataset_min, dataset_max)
testPredict = numpy_minmax_inv(testPredict, dataset_min, dataset_max)
trainY = numpy_minmax_inv(trainY, dataset_min, dataset_max)
testY = numpy_minmax_inv(testY, dataset_min, dataset_max)
print(trainPredict.shape, testPredict.shape)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:, look_f-1], trainPredict))
trainNSE = NSE(trainPredict, trainY[:, look_f-1])
print('Train Score: %.2f RMSE' % (trainScore))
print('Train Nash is:', trainNSE)
testScore = math.sqrt(mean_squared_error(testY[:, look_f-1], testPredict))
testNSE = NSE(testPredict, testY[:, look_f-1])
print('Test Score: %.2f RMSE' % (testScore))
print('Test Nash is:', testNSE)
allX, _ = create_dataset(dataset, look_back, look_f)

all_series_predict = model.predict(allX)

all_series_predict_fine = numpy_minmax_inv(all_series_predict[:, look_f-1], dataset_min, dataset_max)
plt.plot(datasetor[:, 0])
all_series_plot = np.empty_like(dataset)
all_series_plot[:, :] = np.nan
all_series_plot = np.vstack((all_series_plot, all_series_plot[0:look_f, :]))

all_series_plot[look_back + look_f: len(all_series_predict_fine) + look_back + look_f, 0] = all_series_predict_fine
plt.plot(all_series_plot)

plt.show()
