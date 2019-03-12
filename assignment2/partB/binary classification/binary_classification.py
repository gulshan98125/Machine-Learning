import numpy as np
from numpy.linalg import norm
import cPickle as pickle
import cvxopt as cpt  #matrix as cvxopt_matrix
# from cvxopt import solvers as cvxopt_solvers

train_path = '../data/train.csv'
gamma = 0.05
test_path = '../data/test.csv'
C = 1.0
d=5 # entry number last digit
# cpt.solvers.options['abstol'] = 1e-10
# cpt.solvers.options['reltol'] = 1e-10
# cpt.solvers.options['feastol'] = 1e-10

def getKernel(x1,x2):
	return np.exp(-gamma*(norm(x1-x2)**2))

def getKernelMatrix(X):
	m = X.shape[0]
	K = np.zeros((m,m)).tolist()
	for i in xrange(m):
		for j in xrange(m):
			K[i][j] = np.exp(-gamma*(norm(X[i]-X[j])**2))
		print(i)
	return np.array(K)

def train(train_path,C,kernel):
	# reading files and creating X,Y arrays
	# for y, d == -1 and d+1 == 1
	X = []
	Y = []
	with open(train_path,'r') as train_file:
		for line in train_file:
			x_temp = []
			line = line.split('\n')[0]
			line_arr = line.split(',')
			x_part = line_arr[:-1]
			y_part = line_arr[-1]
			if int(y_part)==d:
				Y.append(-1.0)
				for i in x_part:
					x_temp.append(float(i)/255.0)
				X.append(x_temp)
			elif int(y_part)==d+1:
				Y.append(1.0)
				for i in x_part:
					x_temp.append(float(i)/255.0)
				X.append(x_temp)
	X = np.array(X)
	Y = np.array(Y)

	if kernel=='linear':
		m,n = X.shape
		Y = Y.reshape(Y.shape[0],1)
		x_prime = Y*X
		P = cpt.matrix(np.dot(x_prime , x_prime.T))
		q = cpt.matrix(-1.0*np.ones((m, 1)))
		I = np.eye(m) # identity matrix
		G = cpt.matrix(np.vstack(((-1.0)*I,I)))
		h = cpt.matrix(np.hstack((np.zeros(m), C*np.ones(m))))
		A = cpt.matrix(Y.reshape(1, Y.shape[0]))
		b = cpt.matrix(np.zeros(1))
		solution = cpt.solvers.qp(P, q, G, h, A, b)
		alphas = np.array(solution['x'])
		# alphas = np.array(list(set(zip(*alphas)[0]))).reshape(-1,1)
		w = np.dot((Y*alphas).T,X).reshape(-1,1)
		S = (alphas > 1.0e-5).flatten()
		b = Y[S] - np.dot(X[S], w)
		temp = zip(*b)[0]
		b =  sum(temp)/len(temp)
		return (w,b)

	elif kernel=='gaussian':
		# print("calculating kernel matrix")
		# K = getKernelMatrix(X,gamma)
		# with open('kernel.pickle','wb') as outfile:
		# 	pickle.dump(K,outfile,-1)
		# print("done kernel matrix calculation")
		
		with open('kernel.pickle','rb') as infile:
			print("loading kernel matrix")
			K = pickle.load(infile)
			print("done loading kernel matrix")

		Y = Y.reshape(Y.shape[0],1)
		y_yt = np.dot(Y,Y.T)
		P = cpt.matrix(K*y_yt)
		m,n = X.shape
		q = cpt.matrix(-1.0*np.ones((m, 1)))
		I = np.eye(m) # identity matrix
		G = cpt.matrix(np.vstack(((-1.0)*I,I)))
		h = cpt.matrix(np.hstack((np.zeros(m), C*np.ones(m))))
		A = cpt.matrix(Y.reshape(1, Y.shape[0]))
		b = cpt.matrix(np.zeros(1))
		solution = cpt.solvers.qp(P, q, G, h, A, b)
		alphas = np.array(solution['x'])
		# alphas = np.array(list(set(zip(*alphas)[0]))).reshape(-1,1)
		# w = np.dot((Y*alphas).T,X).reshape(-1,1)
		S = (alphas > 1.0e-5).flatten()
		ind = np.arange(len(alphas))[S]

		new_alphas = alphas[S]
		support_vectors = X[S]
		new_Y = Y[S]
		b = 0.0
		temp = new_alphas*new_Y
		print("calculating B...")
		for k in range(len(new_alphas)):
			b += new_Y[k]
			b -= np.sum(temp*K[ind[k],S])
		b /= len(new_alphas)
		return (support_vectors,new_Y,new_alphas,b)

def predict(tuple_val,test_path,kernel):
	totalCount=0
	totalN=0
	totalP=0
	correctN = 0
	correctP = 0
	print("predicting...")
	with open(test_path,'r') as test_file:
		if kernel=='linear':
			(w,b) = tuple_val
			for line in test_file:
				x_temp = []
				line = line.split('\n')[0]
				line_arr = line.split(',')
				x_part = line_arr[:-1]
				y_part = line_arr[-1]
				if int(y_part)==d:
					totalN+=1
					#d==-1
					for i in x_part:
						x_temp.append(float(i)/255.0)
					prediction = np.sign(np.dot(w.T,np.array(x_temp))+b)
					
					if prediction == -1.0:
						correctN+=1
					totalCount+=1
				elif int(y_part)==d+1:
					totalP+=1
					#d+1==1
					for i in x_part:
						x_temp.append(float(i)/255.0)
					prediction = np.sign(np.dot(w.T,np.array(x_temp))+b)
					
					if prediction == 1.0:
						correctP+=1
					totalCount+=1

		elif kernel=='gaussian':
			(support_vectors,y_i,alpha_i,b) = tuple_val
			for line in test_file:
				x_temp = []
				line = line.split('\n')[0]
				line_arr = line.split(',')
				x_part = line_arr[:-1]
				y_part = line_arr[-1]
				if int(y_part)==d:
					totalN+=1
					#d==-1
					for i in x_part:
						x_temp.append(float(i)/255.0)

					X_dot_product = np.zeros(support_vectors.shape[0]).tolist()
					for i in range(support_vectors.shape[0]):
						X_dot_product[i] = getKernel(support_vectors[i],x_temp)
					prediction = np.sign(np.dot((alpha_i*y_i).T ,np.array(X_dot_product)) + b)[0]
					if prediction == -1.0:
						correctN+=1
					totalCount+=1
				elif int(y_part)==d+1:
					totalP+=1
					#d+1==1
					for i in x_part:
						x_temp.append(float(i)/255.0)
					X_dot_product = np.zeros(support_vectors.shape[0]).tolist()
					for i in range(support_vectors.shape[0]):
						X_dot_product[i] = np.dot(support_vectors[i],np.array(x_temp))
					prediction = np.sign(np.dot((alpha_i*y_i).T ,np.array(X_dot_product)) + b)[0]
					if prediction == 1.0:
						correctP+=1
					totalCount+=1
	return (((correctP+correctN)*1.0)/totalCount,correctN,totalN,correctP,totalP)

kernel='linear'
tuplewa=train(train_path,C,kernel)
print(predict(tuplewa,test_path,kernel))
