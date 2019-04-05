import numpy as np
import pandas as pd
import sklearn.metrics as SKL
import time
import sys
from sklearn.model_selection import train_test_split
np.warnings.filterwarnings('ignore')
global global_start_time
global tol 
tol = 10**-6


class layer:
	def __init__(self):
		self.w = None
		self.b = None
		self.activation_function = None
		self.z = None
		self.a = None
		self.params = None

class activation_class:
	def __init__(self,method):
		self.method = method
	def activation(self,y):
		if self.method=='sigmoid':
			return 1.0 / (1 + np.exp(-y))
		elif self.method=='relu':
			y[y < 0] = 0
			return y
		else:
			print("Non supported activation function")
	def activation_der(self,y):
		if self.method == 'sigmoid':
			return self.activation(y) * (1 - self.activation(y))
		elif self.method=='relu':
			y[y < 0] = 0
			y[y > 0] = 1
			return y

class neural_network:
	def __init__(self,num_inputs,hidden_layers,num_outputs,layer_activations,SGD_batch_size,learning_rate,epochs):
		#hidden_layers = [5,10] ,etc
		# layer_activations = ['sigmoid','ReLU'],etc
		self.eeta = learning_rate
		self.epochs = epochs
		self.SGD = SGD_batch_size
		self.num_layers = len(hidden_layers)+2
		self.layer_dict = {}
		self.best_layer = None

		layers = [num_inputs]+hidden_layers+[num_outputs]
		for i in range(len(layers)-1):
			l = layer()
			l.w = np.random.randn(layers[i], layers[i + 1]) / np.sqrt(layers[i])
			l.b = np.random.randn(layers[i + 1])/np.sqrt(layers[i])
			if i!=0:#not first layer
				l.activation_function = activation_class(layer_activations[i-1])
			self.layer_dict[i+1] = l
		#for last layer
		l = layer()
		l.activation_function = activation_class(layer_activations[i])
		self.layer_dict[i+2] = l

	@staticmethod
	def error(y_pred,y_actual):
		return np.nanmean(np.square(y_pred - y_actual))

	def proceed_forward(self,X):
		self.layer_dict[1].a = X
		for i in range(1,self.num_layers):
			self.layer_dict[i+1].z = np.dot(self.layer_dict[i].a, self.layer_dict[i].w) + self.layer_dict[i].b
			self.layer_dict[i+1].a = self.layer_dict[i+1].activation_function.activation(self.layer_dict[i+1].z)
		return self.layer_dict

	def back_propogation(self,layer_dict,Y):
		#output layer delta
		out_layer = layer_dict[self.num_layers]
		y_pred = out_layer.a
		delta = (y_pred- Y)*(out_layer.activation_function.activation_der(y_pred))
		dw = np.dot(layer_dict[self.num_layers - 1].a.T, delta)
		layer_dict[self.num_layers-1].params = [dw,delta]
		reversed_list = list(reversed(range(2, self.num_layers)))
		for i in reversed_list:
			l = layer_dict[i]
			delta = np.dot(delta, l.w.T) * l.activation_function.activation_der(l.z)
			dw = np.dot(layer_dict[i-1].a.T, delta)
			layer_dict[i - 1].params = [dw, delta]

		for j in reversed(range(1,self.num_layers)):
			p = layer_dict[j].params
			self.update_w_and_b(j,p[0], p[1])

	def update_w_and_b(self,index,dw,delta):
		self.layer_dict[index].w -= self.eeta * dw
		self.layer_dict[index].b -= self.eeta * np.nan_to_num(np.nanmean(delta, 0))

	def train(self,X,Y,lr_method):
		last_80_old_epoch_error = 1
		last_consecutive_epoch_error = 1
		min_error = 1
		X,X1,Y,Y1 = train_test_split(X,Y,test_size=0.03)
		for i in range(self.epochs):
			X_new = X
			Y_new = Y
			for j in range(0,x.shape[0],self.SGD):
				l_dict = self.proceed_forward(X_new[j:j+self.SGD])
				self.back_propogation(l_dict, Y_new[j:j+self.SGD])

			l_dict = self.proceed_forward(X1)
			err = self.error(l_dict[self.num_layers].a, Y1)
			print( "error=%f"%(err) )

			if err<min_error:
				self.best_layer = self.layer_dict
				min_error = err

			if (i+1)%80==0:
				if (err >= last_80_old_epoch_error) or (abs(last_80_old_epoch_error-err) <= 10**-6) or ((time.time()-global_start_time)>600):
					print("\nminimum error on validation set=%f"%min_error)
					print("num epochs=%d"%i)
					break
				last_80_old_epoch_error = err

			if (lr_method=='variable') and (i+1)%2==0 and i>30: #basically skips every 2 epochs and then checks
				if last_consecutive_epoch_error - err < tol:
					print("-----eeta reduced-----")
					self.eeta = self.eeta*(0.2)
			last_consecutive_epoch_error = err


	def predict(self,X):
		l_dict = self.proceed_forward(X)
		return np.argmax(l_dict[self.num_layers].a,axis=1)

if __name__ == "__main__":
	path_to_config_file = str(sys.argv[1])
	path_to_one_hot_train = str(sys.argv[2])
	path_to_one_hot_test = str(sys.argv[3])

	with open(path_to_config_file,'r') as f:
		arr = f.readlines()
	num_inputs = int(arr[0].split('\n')[0])
	num_outputs = int(arr[1].split('\n')[0])
	batch_size = int(arr[2].split('\n')[0])
	num_hl_layer = int(arr[3].split('\n')[0])
	temp = arr[4].split('\n')[0].split()
	hl_layer_arr = []
	for i in range(num_hl_layer):
		hl_layer_arr.append(int(temp[i]))

	non_linearity = arr[5].split('\n')[0] #relu or sigmoid
	learning_rate_method = arr[6].split('\n')[0] #fixed or variable

	np.random.seed(1)
	df_train = pd.read_csv(path_to_one_hot_train)
	df_train = df_train.drop(columns=['Unnamed: 0'])
	X = df_train.drop(columns='Y')
	Y = df_train['Y']
	x = np.array(X)
	y = np.array(Y)
	y_hot = np.eye(num_outputs)[y]
	if non_linearity=='sigmoid':
		nn = neural_network(num_inputs,hl_layer_arr,num_outputs,['sigmoid']*(len(hl_layer_arr)+1),SGD_batch_size=batch_size,learning_rate=0.1,epochs=50000)
	else:
		nn = neural_network(num_inputs,hl_layer_arr,num_outputs,(['relu']*(len(hl_layer_arr)))+['sigmoid'],SGD_batch_size=batch_size,learning_rate=0.1,epochs=50000)
	
	start_time,global_start_time = time.time(), time.time()
	nn.train(x,y_hot,learning_rate_method)
	nn.layer_dict = nn.best_layer
	end_time = time.time()
	print("training time = %d seconds"%(end_time-start_time))

	y_train_true = y
	y_train_pred = nn.predict(x)
	print("train accuracy = %f percent" % ((sum(y_train_true==y_train_pred)*100.0)/len(y_train_true)) )

	df_test = pd.read_csv(path_to_one_hot_test)
	df_test = df_test.drop(columns=['Unnamed: 0'])
	X_test = df_test.drop(columns='Y')
	Y_test = df_test['Y']
	x_t = np.array(X_test)
	y_t = np.array(Y_test)

	y_test_true = y_t
	y_test_pred = nn.predict(x_t)

	print("test accuracy = %f percent" % ((sum(y_test_true==y_test_pred)*100.0)/len(y_test_true)) )
	print("test confusion matrix below:")
	print(SKL.confusion_matrix(y_test_true,y_test_pred))
