import xgboost as xgb

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt

rng = np.random.RandomState(31337)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), ]
		dataX.append(a)
	for i in range(len(dataset) - (2 * look_back - 1)):
		b = dataset[(i + look_back):(i + 2 * look_back), 0]
		dataY.append(b)
	return np.array(dataX), np.array(dataY)


# load the dataset
dataframe = pd.read_csv('Caposele-Senerchia-Q-P.csv', usecols=[1, 2], engine='python', skipfooter=3, delimiter=';')
dataframe1 = dataframe
dataframe1.iloc[:, 1:2] = dataframe.iloc[:, 1:2].rolling(90, 1).mean()
dataframe1.iloc[:, 1:2] = dataframe.iloc[:, 1:2].rolling(90, 1).mean()
dataframe1.iloc[:, 1:2] = dataframe.iloc[:, 1:2].rolling(90, 1).mean()
dataset = dataframe1.values
datasetor = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(datasetor)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, ], dataset[train_size:len(dataset), ]

look_back = 60
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = trainX[:len(trainY), :, :]
trainX1 = np.reshape(trainX[:, :, 0], (len(trainX), trainX.shape[1]))
trainX2 = np.reshape(trainX[:, :, 1], (len(trainX), trainX.shape[1]))
# reshape input
# trainY = trainY[:, 0]

testY = testY[:, 0]
testX = testX[:len(testY), :, :]
testX1 = np.reshape(testX[:, :, 0], (len(testX), testX.shape[1]))
testX2 = np.reshape(testX[:, :, 1], (len(testX), testX.shape[1]))

print(trainX1.shape, trainY.shape, testX1.shape, testY.shape)
xgb_model = xgb.XGBRegressor(n_estimators = 100, seed = 42, learning_rate= 0.1, max_depth=2,min_child_weight=10,gamma=0.5,subsample=0.5,colsample_bytree=0.9).fit(trainX1, trainY)
predictions = xgb_model.predict(testX1)
actuals = testY
print(mean_squared_error(actuals, predictions))
plt.plot(predictions)
plt.plot(actuals)
plt.show()

xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
				   {'max_depth': [2, 4, 6],
					'n_estimators': [50, 100, 200]}, verbose=1)
clf.fit(trainX1, trainY)
print(clf.best_score_)
print(clf.best_params_)
