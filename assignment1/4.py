import sys
import math
import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt

def calulate_roots(a,b,c):
	D = b**2-4.0*a*c
	if D==0:
		return (1,(-b)/(2.0*a),0.0)
	elif D<0:
		return (0,0.0,0.0) 
	else:
		r1 = -b + math.sqrt(D)
		r2 = -b - math.sqrt(D)
		return (2,r1/2.0*a,r2/2.0*a)

def get_meus_and_sigmas(X,Y, seperator): # calculates mu0,mu1,sigma0,sigma1 based on seperator value
	
	dimension = len(X[0])
#***************calculating mu0,mu1****************
	mu0 = np.array([[0.0]]*dimension)
	mu1 = np.array([[0.0]]*dimension)
	divisor1 = 0
	divisor2 = 0
	for i in xrange(len(Y)):
		if Y[i]==0:
			mu0 = np.add(mu0,np.transpose([X[i]]))
			divisor1+=1
		if Y[i]==1:
			mu1 = np.add(mu1,np.transpose([X[i]]))
			divisor2+=1
	mu0 = np.multiply(mu0,(1.0/divisor1))
	mu1 = np.multiply(mu1,(1.0/divisor2))
#**************************************************

	if seperator==0:
#---------------------------calculating sigma--------------------------------
		sigma_res = np.zeros((dimension,dimension))
		for i in xrange(len(Y)):
			if(Y[i]==0):
				a = np.subtract(np.transpose([X[i]]),mu0)
				sigma_res = np.add(sigma_res, np.matmul(a,np.transpose(a)))
			else:
				a = np.subtract(np.transpose([X[i]]),mu1)
				sigma_res = np.add(sigma_res, np.matmul(a,np.transpose(a)))
		sigma_res = sigma_res*(1.0/len(Y))
#-----------------------------------------------------------------------------
		return mu0,mu1,sigma_res,sigma_res
	else:
#****************************calculating sigma0 and sigma1*******************************
		sigma0, sigma1 = np.zeros((dimension,dimension)), np.zeros((dimension,dimension))
		count_zeros = 0
		count_ones = 0
		for i in xrange(len(Y)):
			if Y[i]==1:
				count_ones+=1
				xi_minus_mu1 = np.subtract(np.transpose([X[i]]),mu1)
				sigma1 = np.add(sigma1, np.matmul(xi_minus_mu1,np.transpose(xi_minus_mu1)))
			else:
				count_zeros+=1
				xi_minus_mu0 = np.subtract(np.transpose([X[i]]),mu0)
				sigma0 = np.add(sigma0,np.matmul(xi_minus_mu0,np.transpose(xi_minus_mu0)))
		sigma0 = (1.0/count_zeros)*sigma0
		sigma1 = (1.0/count_ones)*sigma1
#*****************************************************************************************
		return mu0,mu1,sigma0,sigma1


if __name__ == "__main__":
	x_name = str(sys.argv[1])
	y_name = str(sys.argv[2])
	seperator = int(sys.argv[3]) # 0 = linear, 1 = quadratic
	
#-----------------------reading files-------------------------------
	x_file = open(x_name, "r")
	y_file = open(y_name, "r")
	X=[]
	Y=[]
	for i in x_file.readlines():
		a = i.split('\n')[0]
		X.append([float(a.split(' ')[0]), float(a.split(' ')[-1])])
	for j in y_file.readlines():
		a = j.split('\n')[0]
		Y.append(int(a=='Alaska'))
	x_file.close()
	y_file.close()
#-------------------------------------------------------------------

#*****************normalizing data********************
	# mean = np.mean(X,axis=0)
	# var = np.var(X,axis=0)

	# for i in xrange(len(Y)):
	# 	X[i][0] = (X[i][0]-mean[0])/math.sqrt(var[0])
	# 	X[i][1] = (X[i][1]-mean[1])/math.sqrt(var[1])
#*****************************************************
	
	x1_min = X[0][0]
	x1_max = X[0][0]
	for i in xrange(len(X)):
		x1_min = min(x1_min,X[i][0])
		x1_max = max(x1_max,X[i][0])
	x = np.linspace(x1_min-1,x1_max+1,40)
	y = []
	mu0,mu1,sigma0,sigma1 = get_meus_and_sigmas(X,Y,seperator)

#-----------calculating coeffs for linear boundary---------------------
	eqn1 = np.matmul(inv(sigma0),np.subtract(mu0,mu1))
	eqn2 = np.matmul(np.transpose(mu1),np.matmul(inv(sigma0),mu1))
	eqn3 = np.matmul(np.transpose(mu0),np.matmul(inv(sigma0),mu0))
	eqn1,eqn2,eqn3 = eqn1[0][0],eqn1[1][0],np.subtract(eqn3,eqn2)[0][0]
#----------------------------------------------------------------------

#****************plotting linear boundary*******************
	for i in x:
		y.append(0.5*(eqn3-(i*eqn1))/eqn2)
	plt.plot(x,y,'k',color='m',label='linear boundary')
	print("mu0",mu0)
	print("mu1",mu1)

	if seperator==0 :
		print("sigma",sigma0)
		plt.plot(x,y,'k',color='m',label='linear boundary')
#***********************************************************
	else:
		print("sigma0",sigma0)
		print("sigma1",sigma1)
		
		
#------------------calculating coefficients for quadratic equation---------------------
		sigma1_inv,sigma0_inv = inv(sigma1), inv(sigma0)
		phi1 = (1.0/len(Y))*sum(Y)
		phi0 = 1-phi1
		a,b,c,d = sigma1_inv[0][0],sigma1_inv[0][1],sigma1_inv[1][0],sigma1_inv[1][1]
		p,q,r,s = sigma0_inv[0][0],sigma0_inv[0][1],sigma0_inv[1][0],sigma0_inv[1][1]
		mu00,mu01,mu10,mu11 =mu0[0][0],mu0[1][0],mu1[0][0],mu1[1][0]
		C = math.log(abs(det(sigma0)/det(sigma1)))+ 2*math.log(phi1/phi0)
		u = d-s
#--------------------------------------------------------------------------------------
		
		x_quadratic = []
		y_quadratic = []
		for i in range(35):
			x_val=x[i]
#-----------------------using quadratic coefficients below to find y values using x-----------------------------
			u = d-s
			v = -2*d*mu11 + 2*s*mu01 + b*x_val - b*mu10 + c*x_val - c*mu10 -q*x_val + q*mu00 - r*x_val + r*mu00
			w = C - a*((x_val-mu10)**2) + p*((x_val-mu00)**2) + (b+c)*mu11*x_val - (q+r)*mu01*x_val \
				- d*(mu11**2) + s*(mu01**2) - (b+c)*mu01*mu11 + (q+r)*mu01*mu00
			roots = calulate_roots(u,v,-w)
			if(roots[0]==1):
				x_quadratic.append(x[i])
				y_quadratic.append(roots[1])
			if(roots[0]==2):
				x_quadratic.append(x[i])
				y_quadratic.append(roots[2])
#----------------------------------------------------------------------------------------------------------------
		plt.plot(x_quadratic,y_quadratic,'b',label='Quadratic Boundary')
#************************plotting alaska and canada*************************
	ones_x1,ones_x2 = [], []
	zeros_x1,zeros_x2 = [], []
	for i in xrange(len(X)):
		if(Y[i]==1):
			ones_x1.append(X[i][0])
			ones_x2.append(X[i][1])
		else:
			zeros_x1.append(X[i][0])
			zeros_x2.append(X[i][1])
	plt.plot(ones_x1,ones_x2,'gx',markerSize=4,color='b',label='Alaska')
	plt.plot(zeros_x1,zeros_x2,'ro',markerSize=4,color='g',label='Canada')
#****************************************************************************
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend()
	plt.show()