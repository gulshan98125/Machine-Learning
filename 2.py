import sys
import numpy as np
from numpy import genfromtxt
from datetime import datetime
from numpy.linalg import inv
import matplotlib.pyplot as plt


def nonweighted_linear_reg(x_values,y_values): # normal lin reg 

	#accomodating multidimension X
	# try:
	# 	x_values.shape(1)
	# 	X = x_values
	# 	X = np.insert(x_values,0,1,axis=1)
	# except:
	# 	X = np.transpose([x_values])
	# 	X = np.insert(X,0,1,axis=1)
	X = np.transpose([x_values])
	X = np.insert(X,0,1,axis=1)
	Y = np.transpose([y_values])

	theta = np.matmul(inv(np.matmul(np.transpose(X),X)), np.matmul(np.transpose(X),Y))
	return (theta[0][0],theta[1][0])

def fun(thet0, thet1, x): # returns h_theta_x
	return thet0 + thet1*x

def weighted_linear_reg(x_values,y_values,x0,tau):
	X = np.transpose([x_values])
	X = np.insert(X,0,1,axis=1) # adding bias to every x in X
	Y = np.transpose([y_values])
	W = np.zeros((len(x_values), len(y_values)))
	W = W.tolist()
	for i in xrange(0,len(x_values)): # generating weight matrix
		superscript = -((x0-x_values[i])**2)/(2*(tau**2))
		W[i][i] = 1.0e1**superscript
	X_T = np.transpose(X)
	try:
		theta = np.matmul(np.matmul(np.matmul(inv(np.matmul(np.matmul(X_T,W), X)), X_T) , W), Y)
		# this function can get singular matrix so ignoring those points in which it gets
	except:
		return None,None

	return fun(theta[0][0], theta[1][0], x0),x0

if __name__ == "__main__":
	x_name = str(sys.argv[1])
	y_name = str(sys.argv[2])
	tau = float(sys.argv[3])
	x_val = genfromtxt(x_name, delimiter=',')
	y_val = genfromtxt(y_name, delimiter=',')

	x = np.linspace(min(x_val)-1,max(x_val)+1,500)
	fig, axs = plt.subplots(2, 1)

	theta_0, theta_1 = nonweighted_linear_reg(x_val,y_val)
	axs[0].plot(x,fun(theta_0, theta_1, x), linewidth=4, color='m')
	axs[0].scatter(x_val,y_val)
	axs[0].set_title("simple linear regression")

	out = []
	inp = []
	for j in x:
		a,b = weighted_linear_reg(x_val,y_val,j,tau)
		if a:
			out.append(a)
			inp.append(b)
	axs[1].scatter(x_val,y_val,linewidths=2)
	axs[1].scatter(inp,out, linewidths=0, color='m')
	axs[1].set_title("weighted linear regression, T = "+str(tau))

	plt.show()


