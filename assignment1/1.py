import numpy as np
import sys
from numpy import genfromtxt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3



def cost_function(theta0,theta1,x_values,y_values):
	res = 0
	for i in xrange(len(y_values)):
		res += (y_values[i] - (theta0+theta1*x_values[i]))**2
	return res*(1.0/(2*len(y_values)))

def batch_gradient_descent(x_values,y_values,neeta_val):
	start_time=datetime.now()
	# theta = np.transpose([[0.0,0.0]])
	theta = [0.0,0.0]
	all_theta0=[0.0]
	all_theta1=[0.0]
	all_costs = [cost_function(0.0,0.0,x_values,y_values)]

	neeta = (neeta_val)*(1.0/(len(x_values)*10.0))
	cost = 0.0
	previous_theta = [0.0,0.0]
	print("please wait, running...")
	while(1):
		previous_theta = theta[:]
		for j in xrange(len(theta)):
			summation = 0
			for i in xrange(len(x_values)):
				X = [1,x_values[i]]
				# h_theta_x = np.dot(theta,X)
				const = y_values[i]-theta[0]-theta[1]*x_values[i]
				summation += const*X[j]
				cost+= const**2
			cost = cost*(1.0/(2*len(y_values)))
			theta[j] = theta[j] + neeta*summation
		all_theta0.append(theta[0])
		all_theta1.append(theta[1])
		all_costs.append(cost_function(theta[0],theta[1],x_values,y_values))
		#stopping criteria below
		if abs(np.amax(np.subtract(theta,previous_theta))) <= stop_val:
			end_time = datetime.now()
			print("runtime="+str((end_time-start_time).total_seconds())+" seconds")
			break
	print("theta0, theta1", theta[0], theta[1])
	return (theta[0],theta[1],all_theta0,all_theta1,all_costs)

def fun(thet0, thet1, x): # theta.T of X
	return thet0 + thet1*x


if __name__ == "__main__":
	x_name = str(sys.argv[1])
	y_name = str(sys.argv[2])
	learning_rate = float(sys.argv[3])
	time_gap = float(sys.argv[4])

	x_val = genfromtxt(x_name, delimiter='\n')
	y_val = genfromtxt(y_name, delimiter='\n')
	stop_val = 1.0e-10

	theta_0, theta_1, all_theta0,all_theta1, all_costs= batch_gradient_descent(x_val,y_val,learning_rate)
	theta0_min, theta0_max, theta1_min, theta1_max = min(all_theta0),max(all_theta0),min(all_theta1),max(all_theta1)

#-----------------------------------plotting line and points-----------------------------------------
	x_for_line = np.linspace(min(x_val)-1,max(x_val)+1,100)
	plt.plot(x_for_line,fun(theta_0, theta_1, x_for_line), linewidth=3, color='m',label='regression line')
	plt.scatter(x_val,y_val, color = 'b',label='data points')
	plt.legend()
	plt.xlabel("acidity of wine")
	plt.ylabel("density of wine")
	
#-----------------------------------------------------------------------------------------------------


	figure = plt.figure()
	x_for_mesh = np.linspace(theta0_min,theta0_max,50)
	y_for_mesh = np.linspace(theta1_min,theta1_max,50)
	X_mesh, Y_mesh = np.meshgrid(x_for_mesh, y_for_mesh)
	z_temp = [cost_function(x,y,x_val,y_val) for x,y in zip(np.ravel(X_mesh), np.ravel(Y_mesh))]
	Z_mesh = np.array(z_temp).reshape(X_mesh.shape)
#***********************************contour below********************************************
	
	# #Angles needed for quiver plot
	# anglesx = np.array(all_theta0)[1:] - np.array(all_theta0)[:-1]
	# anglesy = np.array(all_theta1)[1:] - np.array(all_theta1)[:-1]

	# az = figure.add_subplot(1, 1, 1)
	# az.contour(X_mesh, Y_mesh, Z_mesh, 100, cmap = 'jet')
	# az.quiver(all_theta0[:-1], all_theta1[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .9)


#******************************************************************************************************

#----------------------------------------3d mesh below-------------------------------------------------
	ax = figure.add_subplot(111, projection='3d')
	ax.plot_surface(X_mesh,Y_mesh,Z_mesh)
	graph, = ax.plot([0.0],[0.0],cost_function(0.0,0.0,x_val,y_val),c='m', marker='o',markersize=3)
	def gen_animation(i):
		graph.set_data(all_theta0[:i+2], all_theta1[:i+2])
		graph.set_3d_properties(all_costs[:i+2])
		return graph

	ani = animation.FuncAnimation(figure, gen_animation, interval=time_gap*1000)
#----------------------------------------------------------------------------------------------------------
	plt.show()