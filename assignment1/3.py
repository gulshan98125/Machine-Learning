import sys
import numpy as np
from numpy import genfromtxt
from numpy.linalg import inv
import matplotlib.pyplot as plt

stop_val = 1.0e-17

def main_function(x_values, y_values): # x_values is matrix , y_values is array, main function returns final theta
	X = np.insert(x_values,0,1,axis=1) # adding bias to every x in X
	Y = y_values
	num_inputs = len(X)
	theta = [[0.0],[0.0],[0.0]]
	def sigmoid_function(X,theta):
		return 1.0/(1.0+np.exp(-(np.matmul(np.transpose(theta), X))[0]))

	def get_gradient_Ltheta(): # calculating the gradient array of L(theta)
		res = np.zeros((len(theta),1))
		sigmoid_temp = [sigmoid_function(X[i],theta) for i in xrange(num_inputs)]
		for j in xrange(len(theta)):
			for i in xrange(num_inputs):
				res[j][0] += (Y[i] - sigmoid_temp[i])*X[i][j]
		return res

	def get_Hessian_matrix():
		H = np.zeros((len(theta),len(theta)))
		sigmoid_temp = [sigmoid_function(X[i],theta)*(1-sigmoid_function(X[i],theta)) for i in xrange(num_inputs)]
		for j in xrange(len(theta)):
			for k in xrange(len(theta)):
				for i in xrange(num_inputs):
					H[j][k] -= X[i][j]*X[i][k]*sigmoid_temp[i]
		return H
	
	while(1):
		theta_t_plus1 = np.subtract(theta, np.matmul(inv(get_Hessian_matrix()),get_gradient_Ltheta()))
		if abs(np.amax(np.subtract(theta_t_plus1,theta))) < stop_val:
			break
		theta = theta_t_plus1
	return theta


if __name__ == "__main__":
	x_name = str(sys.argv[1])
	y_name = str(sys.argv[2])
	x_val = genfromtxt(x_name, delimiter=',')
	y_val = genfromtxt(y_name, delimiter=',')

	output_theta = main_function(x_val,y_val)
	print("theta",output_theta)
	#graph plotting starts below
	x1_min = x_val[0][0]
	x1_max = x_val[0][0]
	for i in xrange(len(x_val)):
		x1_min = min(x1_min,x_val[i][0])
		x1_max = max(x1_max,x_val[i][0])
	x = np.linspace(x1_min-1,x1_max+1,100)
	y = []
	for i in x:
		y.append(-(output_theta[0][0]+output_theta[1][0]*i)/output_theta[2][0])
	ones_x1,ones_x2 = [], []
	zeros_x1,zeros_x2 = [], []
	for i in xrange(len(y_val)):
		if(y_val[i]==1):
			ones_x1.append(x_val[i][0])
			ones_x2.append(x_val[i][1])
		else:
			zeros_x1.append(x_val[i][0])
			zeros_x2.append(x_val[i][1])
	plt.plot(ones_x1,ones_x2,'gx',markerSize=4,color='b',label='1')
	plt.plot(zeros_x1,zeros_x2,'ro',markerSize=4,color='g',label='0')
	plt.plot(x,y,'k',linewidth=3,color='m',label='decision boundary')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend()
	plt.show()

