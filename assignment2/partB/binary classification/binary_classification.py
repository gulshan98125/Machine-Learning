import numpy as np
from numpy.linalg import norm
import cPickle as pickle
import cvxopt as cpt  #matrix as cvxopt_matrix
# from cvxopt import solvers as cvxopt_solvers

train_path = '../data/train.csv'
test_path = '../data/test.csv'
C = 1.0
d=5 # entry number last digit
# cpt.solvers.options['abstol'] = 1e-10
# cpt.solvers.options['reltol'] = 1e-10
# cpt.solvers.options['feastol'] = 1e-10

def getKernelMatrix(X,gamma):
	m = X.shape[0]
	K = np.zeros((m,m)).tolist()
	for i in xrange(m):
		for j in xrange(m):
			K[i][j] = np.exp(-gamma*norm(X[i]-X[j]))
		print(i)
	return np.array(K)

def train(train_path,C,kernel,gamma):
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
					x_temp.append(float(i))
				X.append(x_temp)
			elif int(y_part)==d+1:
				Y.append(1.0)
				for i in x_part:
					x_temp.append(float(i))
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
		S = (alphas > 1.0e-9).flatten()
		b = Y[S] - np.dot(X[S], w)
		temp = zip(*b)[0]
		b =  sum(temp)/len(temp)
		support_vectors = X[S]
		new_Y_i = Y[S]
		new_alpha_i = alphas[S]
		return support_vectors,new_Y_i,new_alpha_i,b

	elif kernel=='gaussian':
		print("calculating kernel matrix")
		# K = getKernelMatrix(X,gamma)
		# with open('kernel.pickle','wb') as outfile:
		# 	pickle.dump(K,outfile,-1)
		with open('kernel.pickle','rb') as infile:
			K = pickle.load(infile)
		
		print("done kernel matrix calculation")
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
		w = np.dot((Y*alphas).T,X).reshape(-1,1)
		S = (alphas > 1.0e-9).flatten()
		b = Y[S] - np.dot(X[S], w)
		print(b.shape)
		temp = zip(*b)[0]
		b =  sum(temp)/len(temp)
		support_vectors = X[S]
		new_Y_i = Y[S]
		new_alpha_i = alphas[S]
		return support_vectors,new_Y_i,new_alpha_i,b

def predict(support_vectors,y_i,alpha_i,b,test_path):
	totalCount=0
	correctCount=0
	totalN=0
	totalP=0
	correctN = 0
	correctP = 0
	
	with open(test_path,'r') as test_file:
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
					x_temp.append(float(i))

				X_dot_product = np.zeros(support_vectors.shape[0]).tolist()
				for i in range(support_vectors.shape[0]):
					X_dot_product[i] = np.dot(support_vectors[i],np.array(x_temp))
				prediction = np.sign(np.dot((alpha_i*y_i).T ,np.array(X_dot_product)) + b)
				# prediction = np.sign(np.dot(w.T,np.array(x_temp))+b)
				# print(np.dot(w.T,np.array(x_temp))+b,y_part,prediction)
				if prediction == -1.0:
					correctCount+=1
					correctN+=1
				totalCount+=1
			elif int(y_part)==d+1:
				totalP+=1
				#d+1==1
				for i in x_part:
					x_temp.append(float(i))
				X_dot_product = np.zeros(support_vectors.shape[0]).tolist()
				for i in range(support_vectors.shape[0]):
					X_dot_product[i] = np.dot(support_vectors[i],np.array(x_temp))
				prediction = np.sign(np.dot((alpha_i*y_i).T ,np.array(X_dot_product)) + b)
				# print(np.dot(w.T,np.array(x_temp))+b,y_part,prediction)
				if prediction == 1.0:
					correctCount+=1
					correctP+=1
				totalCount+=1
	return ((correctCount*1.0)/totalCount,correctN,totalN,correctP,totalP)


support_vectors,y_i,alpha_i,b=train(train_path,C,'linear',0.05)
# print(support_vectors,support_vectors.shape)
# print(y_i,y_i.shape)
# print(alpha_i,alpha_i.shape)
print(predict(support_vectors,y_i,alpha_i,b,test_path))
print support_vectors.shape


# a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0784313725490196, 0.807843137254902, 0.9372549019607843, 0.2196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0784313725490196, 0.5686274509803921, 0.996078431372549, 0.996078431372549, 0.1803921568627451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5607843137254902, 0.996078431372549, 0.9450980392156862, 0.4, 0.011764705882352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47058823529411764, 0.9921568627450981, 0.9450980392156862, 0.27058823529411763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3607843137254902, 0.9529411764705882, 0.996078431372549, 0.5137254901960784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666666666666667, 0.8352941176470589, 0.996078431372549, 0.6, 0.011764705882352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00784313725490196, 0.6784313725490196, 0.996078431372549, 0.7529411764705882, 0.0784313725490196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3568627450980392, 0.996078431372549, 0.9725490196078431, 0.28627450980392155, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20784313725490197, 0.9294117647058824, 0.9921568627450981, 0.5098039215686274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050980392156862744, 0.7568627450980392, 0.996078431372549, 0.7215686274509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3686274509803922, 0.996078431372549, 0.9764705882352941, 0.30980392156862746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6941176470588235, 0.996078431372549, 0.592156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18823529411764706, 0.9254901960784314, 0.996078431372549, 0.2627450980392157, 0.0, 0.08627450980392157, 0.24313725490196078, 0.09411764705882353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5137254901960784, 0.996078431372549, 0.9019607843137255, 0.027450980392156862, 0.2823529411764706, 0.7019607843137254, 0.996078431372549, 0.9333333333333333, 0.5568627450980392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6941176470588235, 0.996078431372549, 0.592156862745098, 0.4196078431372549, 0.9568627450980393, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.9607843137254902, 0.5607843137254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6941176470588235, 0.996078431372549, 0.6431372549019608, 0.8705882352941177, 0.996078431372549, 0.8274509803921568, 0.2901960784313726, 0.4, 0.996078431372549, 0.788235294117647, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5137254901960784, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.9254901960784314, 0.792156862745098, 0.7294117647058823, 0.996078431372549, 0.7529411764705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411764705882355, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.996078431372549, 1.0, 0.996078431372549, 0.9764705882352941, 0.2980392156862745, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.050980392156862744, 0.7294117647058823, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.8784313725490196, 0.34901960784313724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043137254901960784, 0.615686274509804, 0.996078431372549, 0.996078431372549, 0.7568627450980392, 0.5450980392156862, 0.11764705882352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# print(len(a))