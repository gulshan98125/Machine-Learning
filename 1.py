import numpy as np
from numpy import genfromtxt
from datetime import datetime
import matplotlib.pyplot as plt

x_val = genfromtxt('data/linearX.csv', delimiter='\n')
y_val = genfromtxt('data/linearY.csv', delimiter='\n')
def batch_gradient_descent(x_values,y_values):
	start_time=datetime.now()
	# theta = np.transpose([[0.0,0.0]])
	theta = [0.0,0.0]
	neeta = 0.0295*(1.0/len(x_values))
	cost = 0.0

	print("please wait, running...")
	while(1): 
		previous_cost = cost
		for j in xrange(len(theta)):
			summation = 0
			for i in xrange(len(x_values)):
				X = [1,x_values[i]]
				h_theta_x = np.dot(theta,X)
				summation += (y_values[i]-h_theta_x)*X[j]
			theta[j] = theta[j] + neeta*summation

		# calculating cost/error below
		for i in xrange(len(x_values)):
			X = [1,x_values[i]]
			h_theta_x = np.dot(theta,X)
			cost+= (y_values[i]-h_theta_x)**2
		cost = (cost*1.0)/(2*len(x_values))
		# print(theta,cost)
		
		#stopping criteria below
		if(abs(cost-previous_cost) <= 1.0e-18):
			# print("final theta and cost", theta, cost)
			end_time = datetime.now()
			print("runtime="+str((end_time-start_time).total_seconds())+" seconds")
			break
	print("theta0, theta1", theta[0], theta[1])
	return (theta[0],theta[1])

def fun(thet0, thet1, x):
	return thet0 + thet1*x

theta_0, theta_1 = batch_gradient_descent(x_val,y_val)
x = np.linspace(min(x_val)-5,max(x_val)+5,100)
plt.plot(x,fun(theta_0, theta_1, x), linewidth=3)
plt.scatter(x_val,y_val)
plt.show()