import numpy as np
np.random.seed(100)
class layer:
	def __init__(self, num_inputs,num_neurons, activation_function):
		self.num_neurons = num_neurons
		self.weights = np.random.rand(num_inputs, num_neurons)
		self.accumulated_weights = np.zeros((num_inputs,num_neurons))
		self.activation_function = activation_function
		self.activation_val = None
		self.bias = np.random.rand(num_neurons)
		self.error = None
		self.delta = None

	def layer_activate(self,inp):
		res = np.dot(inp,self.weights) + self.bias
		if self.activation_function=='sigmoid':
			self.activation_val = 1.0/(1 + np.exp(-res))
		elif self.activation_function=='ReLU':
			func = np.vectorize(lambda x: max(0,x))
			self.activation_val = func(res)
		return self.activation_val

class neural_net:
	def __init__(self, method,num_inputs, hidden_layers_list, num_outputs, SGD_batch_size=None,learning_rate=0.001, epochs=100000):
		self.SGD = SGD_batch_size
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.layers_list = []

		l_helper = [num_inputs]+hidden_layers_list
		output_layer = layer(hidden_layers_list[-1],num_outputs,method)
		for i in range(1,len(l_helper)):
			self.layers_list.append(layer(l_helper[i-1],l_helper[i],method))
		self.layers_list.append(output_layer)


	def calculate_output(self,inp):
		res = inp
		for l in self.layers_list:
			res = l.layer_activate(res)
		return res

	def predict(self, X):
		ff = self.calculate_output(X)
		return (ff>0.5)*1


	def activation_derivative(self, method, inp):
		if method=='sigmoid':
			return inp*(1-inp)
		elif method=='ReLU':
			func = np.vectorize(lambda x: 1 if x>0 else 0)
			return func(inp)

	def batch_back_propogate(self,X,Y):
		for k in range(len(X)):
			out = self.calculate_output(X[k])
			reverse = list(reversed(range(len(self.layers_list))))
			for i in reverse:
				l = self.layers_list[i]
				if l==self.layers_list[len(self.layers_list)-1]: #means output layer
					l.error = (Y[k] - out)
					l.delta = l.error * self.activation_derivative(l.activation_function, out)
				else:
					l.error = np.dot(self.layers_list[i+1].weights, self.layers_list[i+1].delta)
					l.delta = l.error * self.activation_derivative(l.activation_function, l.activation_val)

			for j in range(len(self.layers_list)):
				l = self.layers_list[j]
				if j==0:
					curr_input = np.atleast_2d(X[k])
				else:
					curr_input = np.atleast_2d(self.layers_list[j - 1].activation_val)
				l.accumulated_weights += l.delta*curr_input.T
				# print(l.accumulated_weights[1],j)
					# print("--",self.learning_rate*l.delta*curr_input.T)
		
		for m in range(len(self.layers_list)):
			l = self.layers_list[m]
			l.weights += self.learning_rate*l.accumulated_weights*(1.0/len(X))
			l.accumulated_weights=l.accumulated_weights*0.0
	def back_propogate(self,X,Y):
			out = self.calculate_output(X)
			reverse = list(reversed(range(len(self.layers_list))))
			for i in reverse:
				l = self.layers_list[i]
				if l==self.layers_list[len(self.layers_list)-1]: #means output layer
					l.error = (Y - out)
					l.delta = l.error * self.activation_derivative(l.activation_function, out)
				else:
					l.error = np.dot(self.layers_list[i+1].weights, self.layers_list[i+1].delta)
					l.delta = l.error * self.activation_derivative(l.activation_function, l.activation_val)

			for j in range(len(self.layers_list)):
				l = self.layers_list[j]
				if j==0:
					curr_input = np.atleast_2d(X)
				else:
					curr_input = np.atleast_2d(self.layers_list[j - 1].activation_val)
				l.weights += self.learning_rate*l.delta*curr_input.T

	def train(self,X,Y):
		if self.SGD==None:
			for i in range(self.epochs):
				for j in range(len(X)):
					self.back_propogate(X[j],Y[j])
				if i%200==0:
					error = np.mean(np.square(Y - self.calculate_output(X)))
					print("error = %f"%error)
		else:
			# prev_error = 1
			for i in range(self.epochs):
				for j in range(0,len(X),self.SGD):
					self.batch_back_propogate(X[j:j+self.SGD],Y[j:j+self.SGD])
				if i%200==0:
					error = np.mean(np.square(Y - self.calculate_output(X)))
					# if abs(prev_error-error)<10**-3:
					# 	self.learning_rate=self.learning_rate*(0.5)
					# prev_error=error
					print("error = %f"%error)

a = neural_net('sigmoid',4,[2,4,2],4,4,0.2)
X = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
y = np.array([[1,0,0,1],[0,1,0,1],[1,0,1,0],[1,0,0,1]])

a.train(X,y)

print a.predict(X)