import numpy as np
from numpy.linalg import norm
import cPickle as pickle
import time
import sys
import cvxopt as cpt
import svmutil as SVM
from scipy.spatial.distance import pdist,cdist, squareform
import scipy as sp
SVM.svm_model.predict = lambda self, x: SVM.svm_predict([0], [x], self, "-q")[0][0]

# train_path = '../data/train.csv'
gamma = 0.05
# test_path = '../data/test.csv'
C = 1.0
d=5 # entry number last digit
# cpt.solvers.options['abstol'] = 1e-10
# cpt.solvers.options['reltol'] = 1e-10
# cpt.solvers.options['feastol'] = 1e-10
cpt.solvers.options['show_progress'] = False

def vector_prediction(X,x1,alpha_i,y_i,b): # vectorized prediction for gaussian
	pairwise_dists = cdist(X,[x1],'sqeuclidean')
	k = sp.exp(-gamma*(pairwise_dists))
	res = np.sum(alpha_i*y_i*k)+b
	if res==0:
		return -1.0
	else:
		return np.sign(res)

def getKernel(x1,x2):
	return np.exp(-gamma*(norm(x1-x2)**2))

def linearKM(X):
	m = X.shape[0]
	K = np.zeros((m,m)).tolist()
	for i in xrange(m):
		for j in xrange(m):
			K[i][j] = np.dot(X[i],X[j])
		# print(i)
	return np.array(K)

def gaussianKM(X):
	m = X.shape[0]
	K = np.zeros((m,m)).tolist()
	for i in xrange(m):
		for j in xrange(m):
			K[i][j] = np.exp(-gamma*(norm(X[i]-X[j])**2))
		# print(i)
	return np.array(K)

def readFile(file_path):
	# returns X and Y from the file
	# for y, d == -1 and d+1 == 1
	X = []
	Y = []
	with open(file_path,'r') as file:
		for line in file:
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
	return (X,Y)

def train(X,Y,C,kernel):
	if kernel=='linear':
		# print("calculating kernel matrix...")
		# K = linearKM(X)

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
		print("training...")
		start_time = time.time()
		solution = cpt.solvers.qp(P, q, G, h, A, b)
		end_time = time.time()
		print("training time = %s seconds"%(end_time - start_time))
		alphas = np.array(solution['x'])
		# alphas = np.array(list(set(zip(*alphas)[0]))).reshape(-1,1)
		w = np.dot((Y*alphas).T,X).reshape(-1,1)
		S = (alphas > 1.0e-5).flatten()
		
		ind = np.arange(len(alphas))[S]
		new_alphas = alphas[S]
		print("numSV = "+str(len(new_alphas)))
		new_Y = Y[S]
		b = 0.0
		temp = new_alphas*new_Y
		# for k in range(len(new_alphas)):
		# 	b += new_Y[k][0]
		# 	b -= np.sum(temp*K[ind[k],S])
		# b /= len(new_alphas)
		b = (Y[S] - np.dot(X[S], w))[0][0]
		print("b = "+'%.4f'%(b))
		return (w,b)

	elif kernel=='gaussian':
		print("calculating kernel matrix...")
		pairwise_dists = squareform(pdist(X, 'sqeuclidean'))
		K = sp.exp(-gamma*(pairwise_dists))
		# K = gaussianKM(X)
		# with open('kernel.pickle','wb') as outfile:
		# 	pickle.dump(K,outfile,-1)
		Y = Y.reshape(Y.shape[0],1)
		y_yt = np.dot(Y,Y.T)
		P = cpt.matrix(y_yt*K)
		m,n = X.shape
		q = cpt.matrix(-1.0*np.ones((m, 1)))
		I = np.eye(m) # identity matrix
		G = cpt.matrix(np.vstack(((-1.0)*I,I)))
		h = cpt.matrix(np.hstack((np.zeros(m), C*np.ones(m))))
		A = cpt.matrix(Y.reshape(1, Y.shape[0]))
		b = cpt.matrix(np.zeros(1))
		print("training...")
		start_time = time.time()
		solution = cpt.solvers.qp(P, q, G, h, A, b)
		end_time = time.time()
		print("training time = %s seconds"%(end_time - start_time))
		alphas = np.array(solution['x'])
		S = (alphas > 1.0e-5).flatten()
		ind = np.arange(len(alphas))[S]
		new_alphas = alphas[S]
		print("numSV = "+str(len(new_alphas)))
		support_vectors = X[S]
		new_Y = Y[S]
		# b = 0.0
		temp = new_alphas*new_Y
		print("calculating b...")
		b = Y - np.matmul(K,np.array(alphas)*Y)
		b = np.sum(b)/len(b)
		# for k in range(len(new_alphas)):
		# 	b += new_Y[k][0]
		# 	b -= np.sum(temp*K[ind[k],S])
		# b /= len(new_alphas)
		
		print("b = "+'%.4f'%(b))

		return (support_vectors,new_Y,new_alphas,b)

	elif kernel=='libsvm-linear':
		#0=linear, 2=radial
		print("training...")
		start_time = time.time()
		model = SVM.svm_train( Y, X, "-s 0 -c "+str(C)+" -t 0 -q")
		end_time = time.time()
		print("training time = %s seconds"%(end_time - start_time))
		print("b = %s"%(-model.rho[0]))
		print("numSV = %s"%(len(model.get_SV())))
		return model
	elif kernel=='libsvm-gaussian':
		#0=linear, 2=radial
		print("training...")
		start_time = time.time()
		model = SVM.svm_train( Y, X, "-s 0 -c "+str(C)+" -t 2 -q -g "+str(gamma))
		end_time = time.time()
		print("training time = %s seconds"%(end_time - start_time))
		print("b = %s"%(-model.rho[0]))
		print("numSV = %s"%(len(model.get_SV())))
		# print(model.get_SV())
		return model

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

					# X_dot_product = np.zeros(support_vectors.shape[0]).tolist()
					# for i in range(support_vectors.shape[0]):
					# 	X_dot_product[i] = alpha_i[i][0]*y_i[i][0]*getKernel(support_vectors[i],x_temp)
					prediction = vector_prediction(support_vectors,x_temp,alpha_i,y_i,b)
					# prediction = np.sign(np.dot((alpha_i*y_i).T ,np.array(X_dot_product)) + b)[0]
					# prediction = np.sign(np.sum(X_dot_product)+b)
					if prediction == -1.0:
						correctN+=1
					totalCount+=1
				elif int(y_part)==d+1:
					totalP+=1
					#d+1==1
					for i in x_part:
						x_temp.append(float(i)/255.0)
					# X_dot_product = np.zeros(support_vectors.shape[0]).tolist()
					# for i in range(support_vectors.shape[0]):
					# 	X_dot_product[i] = alpha_i[i][0]*y_i[i][0]*getKernel(support_vectors[i],x_temp)
					prediction = vector_prediction(support_vectors,x_temp,alpha_i,y_i,b)
					# prediction = np.sign(np.dot((alpha_i*y_i).T ,np.array(X_dot_product)) + b)[0]
					# prediction = np.sign(np.sum(X_dot_product)+b)
					if prediction == 1.0:
						correctP+=1
					totalCount+=1
		elif kernel == 'libsvm-linear' or kernel=='libsvm-gaussian':
			model = tuple_val
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
					prediction = model.predict(x_temp)
					if prediction == -1.0:
						correctN+=1
					totalCount+=1
				elif int(y_part)==d+1:
					totalP+=1
					#d+1==1
					for i in x_part:
						x_temp.append(float(i)/255.0)
					prediction = model.predict(x_temp)
					if prediction == 1.0:
						correctP+=1
					totalCount+=1

	print("Accuracy = %s %%"%(((correctP+correctN)*100.0)/totalCount))
	print("correct %s = %s, total %s = %s"%(d,correctN,d,totalN))
	print("correct %s = %s, total %s = %s"%(d+1,correctP,d+1,totalP))



if __name__ == "__main__":
	train_data_path = str(sys.argv[1])
	test_data_path = str(sys.argv[2])
	part_num = str(sys.argv[3])
	X,Y = readFile(train_data_path)

	if part_num=="a":
		kernel='linear'
		tuplewa=train(X,Y,C,kernel)
		predict(tuplewa,test_data_path,kernel)
	elif part_num=="b":
		kernel='gaussian'
		tuplewa=train(X,Y,C,kernel)
		predict(tuplewa,test_data_path,kernel)
	elif part_num=="c":
		k1 = 'libsvm-linear'
		k2 = 'libsvm-gaussian'
		print("--------linear kernel--------")
		tuplewa=train(X,Y,C,k1)
		predict(tuplewa,test_data_path,k1)
		print("-----------------------------")
		print("********gaussin kernel********")
		tuplewa=train(X,Y,C,k2)
		predict(tuplewa,test_data_path,k2)
		print("*****************************")
	
	

