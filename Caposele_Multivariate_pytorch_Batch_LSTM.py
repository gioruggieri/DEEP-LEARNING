
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
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
	nom = torch.sum(residuals**2)
	den = torch.sum((y - torch.mean(y_pred))**2)
	OF = 1. - nom/den

	if optim == True:
		OF = 1. - OF
	return OF

def root_mean_sq_error(y_pred, y):
	y_pred = numpy_minmax_inv(y_pred.data.cpu().numpy(), dataset_min, dataset_max)
	y = numpy_minmax_inv(y.data.cpu().numpy(), dataset_min, dataset_max)
	return math.sqrt(mean_squared_error(y, y_pred))

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
dataframe = pd.read_csv('Caposele-Senerchia-Q-P.csv', usecols=[1, 2], engine='python', skipfooter=3, delimiter=';')
dataframe1 = dataframe
dataset = dataframe1.values.astype('float32')
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

# train1 = scaler1.fit_transform(train[:,0])
# train2 = scaler2.fit_transform(train[:,1])
# train = np.column_stack((train1,train2))
#
# test1 = scaler1.transform(test[:,0])
# test2 = scaler2.transform(test[:,1])
# test = np.column_stack((test1,test2))

# reshape into X=t and Y=t+1
look_back = 60
look_f = 60
trainX, trainY = create_dataset(train, look_back, look_f)
testX, testY = create_dataset(test, look_back, look_f)
trainX = trainX[:len(trainY), :]
testX = testX[:len(testY), :]
trainX = trainX.reshape(len(trainX), look_back, 2, order='F')
testX = testX.reshape(len(testX), look_back, 2, order='F')
print('trainY is:', trainY.shape)
print('datasetor is :', datasetor.shape, 'trainX is: ', trainX.shape, 'testX is: ', testX.shape, 'testY is:',
	  testY.shape, 'traindataset is', dataset[train_size:len(dataset), ].shape)

trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).float()
testX = torch.from_numpy(testX).float()
testY = torch.from_numpy(testY).float()
# trainX = Variable(trainX.float())
# trainY = Variable(trainY.float(), requires_grad=False)
if torch.cuda.is_available():
	trainX = trainX.cuda()
	trainY = trainY.cuda()
	testX = testX.cuda()
	testY = testY.cuda()
batch = 32
train = data_utils.TensorDataset(trainX, trainY)
train_loader = data_utils.DataLoader(train, batch_size=batch, shuffle=True)
test = data_utils.TensorDataset(testX, testY)
test_loader = data_utils.DataLoader(test, batch_size=batch, shuffle=True)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.LSTM(2, look_back, 1, batch_first=True).cuda()

is_cuda = torch.cuda.is_available()
if is_cuda:
	model.cuda()

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
nash_crit = NSE
loss_fn = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters())
batch = 32

loss_test_list = []
loss_epoch = 0
n_epoch_loss = []
n_test_loss = []
def train(epoch):
	model.train()
	loss_train = 0
	nash_train = 0
	rmse_train = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		if len(data) != batch:
			continue
		data, target = Variable(data), Variable(target)
		# Instantiating new c0 and h0 on every forward...I hope
		c0 = Variable(torch.rand(1, batch, look_back).cuda(), requires_grad=False)
		h0 = Variable(torch.rand(1, batch, look_back).cuda(), requires_grad=False)
		optimizer.zero_grad()
		output, hn = model(data, (h0, c0))
		loss = loss_fn(output[:, -1], target[:, -1])
		loss.backward()
		optimizer.step()
		if batch_idx % 1 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, (1 + batch_idx) * len(data), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.data[0]))
		loss_train += loss.data[0]
		nash_train += nash_crit(output[:, -1], target[:, -1])
		rmse_train += root_mean_sq_error(output[:, -1], target[:, -1])
	epoch_loss = loss_train/(int(len(train_loader.dataset))/batch)
	epoch_nash = nash_train/(int(len(train_loader.dataset))/batch)
	epoch_rmse = rmse_train/(int(len(train_loader.dataset))/batch)
	n_epoch_loss.append(epoch_loss)
	print('Summary Train Loss:')
	print('Train_Loss is:', math.sqrt(epoch_loss))
	print('Train Nash is:', epoch_nash.data[0])
	print('Train RMSE is:', epoch_rmse)
	print(' ')

def test():
	model.eval()
	test_loss = 0
	nash_test = 0
	rmse_test = 0
	# for data, target in test_loader:
	# 	data, target = Variable(data, volatile=True), Variable(target)
	data, target = Variable(testX, volatile=True), Variable(testY)
		# output = model(data)
	output = model(data)
		# test_loss = loss_fn(output, target).data[0]  # sum up batch loss
	test_loss = loss_fn(output[:, -1], target[:, -1]).data[0]  # sum up batch loss
	nash_test = nash_crit(output[:, -1], target[:, -1]).data[0]
	rmse_test = root_mean_sq_error(output[:, -1], target[:, -1])
	# loss_test = test_loss/(int(len(test_loader.dataset))/batch)
	print('Summary Test Loss:')
	print('Test Loss is:', math.sqrt(test_loss))
	print('Nash TestLoss is:', nash_test)
	print('RMSE TestLoss is:', rmse_test)
	n_test_loss.append(test_loss)

for epoch in range(10):
	train(epoch)
	test()
torch.save(model, 'model.pkl')
plt.plot(n_epoch_loss)
plt.plot(n_test_loss, color='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# allX1 = numpy_minmax(dataset[:, 0], train_min, train_max)
# allX2 = numpy_minmax(dataset[:, 1], train1_min, train1_max)
# allX = np.column_stack((allX1, allX2))
allX, _ = create_dataset(dataset, look_back, look_f)
allX = torch.from_numpy(allX).float()
allX = Variable(allX.float(), requires_grad=False)
if torch.cuda.is_available():
	allX = allX.cuda()
all_series_predict = model.forward(allX)
all_series_predict = all_series_predict.data.cpu().numpy()
all_series_predict_fine = numpy_minmax_inv(all_series_predict, dataset_min, dataset_max)
all_series_predict_fine = all_series_predict_fine[:, look_f-1]

print('all_series_predict shape is:', len(all_series_predict), len(all_series_predict[0]))
plt.plot(datasetor[:, 0])
all_series_plot = np.empty_like(dataset)
all_series_plot[:, :] = np.nan
all_series_plot = np.vstack((all_series_plot, all_series_plot[0:look_f, :]))

all_series_plot[look_back + look_f: len(all_series_predict_fine) + look_back + look_f, 0] = all_series_predict_fine
plt.plot(all_series_plot)
plt.show()
