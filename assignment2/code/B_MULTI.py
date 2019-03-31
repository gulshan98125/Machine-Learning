import numpy as np
from numpy.linalg import norm
import cPickle as pickle
import time
import sys
import csv
import itertools
import cvxopt as cpt
import svmutil as SVM
from scipy.spatial.distance import pdist,cdist, squareform
import scipy as sp
from sklearn.model_selection import train_test_split
SVM.svm_model.predict = lambda self, x: SVM.svm_predict([0], [x], self, "-q")

# train_path = '../data/train.csv'
gamma = 0.05
# test_path = '../data/test.csv'
C = 1.0
# cpt.solvers.options['abstol'] = 1e-10
# cpt.solvers.options['reltol'] = 1e-10
# cpt.solvers.options['feastol'] = 1e-10
cpt.solvers.options['show_progress'] = False

def vector_prediction(support_vectors,x1,alpha_i,y_i,b): # vectorized prediction for gaussian
	pairwise_dists = cdist(support_vectors,[x1],'sqeuclidean')
	k = sp.exp(-gamma*(pairwise_dists))
	result = np.sum(alpha_i*y_i*k)+b
	if result==0:
		return -1.0,0
	else:
		return np.sign(result),abs(result)

def abs_score_prediction(support_vectors,x1,alpha_i,y_i,b):
	pairwise_dists = cdist(support_vectors,[x1],'sqeuclidean')
	k = sp.exp(-gamma*(pairwise_dists))
	return abs(np.sum(alpha_i*y_i*k)+b)

def readFile(file_path,d,e):
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
			elif int(y_part)==e:
				Y.append(1.0)
				for i in x_part:
					x_temp.append(float(i)/255.0)
				X.append(x_temp)
	X = np.array(X)
	Y = np.array(Y)
	return (X,Y)

def libsvm_train(X,Y,gamma,C):
	model = SVM.svm_train( Y, X, "-s 0 -c "+str(C)+" -t 2 -q -g "+str(gamma))
	return model

def libsvm_predict(model,x_vector,d,e):
	p = model.predict(x_vector)
	if p[0][0]==-1:
		return d,abs(p[2][0][0])
	else:
		return e,abs(p[2][0][0])

def cvx_train(X,Y): #gaussian cvxopt train, Y is array of 1s and -1s
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
	start_time = time.time()
	solution = cpt.solvers.qp(P, q, G, h, A, b)
	end_time = time.time()
	alphas = np.array(solution['x'])
	S = (alphas > 1.0e-5).flatten()
	ind = np.arange(len(alphas))[S]
	new_alphas = alphas[S]
	support_vectors = X[S]
	new_Y = Y[S]
	temp = new_alphas*new_Y
	b = Y - np.matmul(K,np.array(alphas)*Y)
	b = np.sum(b)/len(b)
	return (support_vectors,new_Y,new_alphas,b)


def cvx_predict(tuple_val,x_vector,d,e):
	(support_vectors,y_i,alpha_i,b) = tuple_val
	p = vector_prediction(support_vectors,x_vector,alpha_i,y_i,b)
	if p[0]==-1:
		return d,p[1]
	else:
		return e,p[1]

def generateAllClassifiers(train_path,method,gamma,C):
	print("generating all Kc2 classifiers...")
	all_digits = np.arange(10)
	all_combinations = list(itertools.combinations(all_digits,2))
	models_list = []
	if method=='cvxopt':
		for comb in all_combinations:
			X,Y = readFile(train_path,comb[0],comb[1])
			tup = cvx_train(X,Y)
			models_list.append(tup)
		with open('models_cvxopt.pickle','wb') as outfile:
			pickle.dump(models_list,outfile,-1)
	else:
		for comb in all_combinations:
			X,Y = readFile(train_path,comb[0],comb[1])
			model = libsvm_train(X,Y,gamma,C)
			models_list.append(model)
			# with open('models_libsvm.pickle','wb') as outfile:
			# 	pickle.dump(models_list,outfile,-1)
	print("done generating all Kc2 classifiers...")
	return models_list

# for last part
def generate_validation_and_test_files(file_path):
	X = []
	Y = []
	with open(file_path,'r') as file:
		for line in file:
			x_temp = []
			line = line.split('\n')[0]
			line_arr = line.split(',')
			x_part = line_arr[:-1]
			y_part = line_arr[-1]
			Y.append(int(y_part))
			for i in x_part:
				x_temp.append(int(i))
			X.append(x_temp)
	X = np.array(X)
	Y = np.array(Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.10,random_state=41)
	train_list = np.hstack((X_train,Y_train.reshape(-1,1)))
	test_list = np.hstack((X_test,Y_test.reshape(-1,1)))
	with open("train_part.csv", 'wb') as myfile:
		wr = csv.writer(myfile)
		for l in train_list:
			wr.writerow(l)
	with open("test_part.csv", 'wb') as myfile:
		wr = csv.writer(myfile)
		for l in test_list:
			wr.writerow(l)

def validation_method(train_path,test_path):
	#first generate train and test files
	generate_validation_and_test_files(train_path)
	c_values = [1.0e-5,1.0e-3,1.0,5.0,10.0]
	validation_accuracy = []
	test_set_accuracy = []
	for c in c_values:
		models_list = generateAllClassifiers("train_part.csv",'libsvm',0.05,c)
		validation_accuracy.append(get_accuracy_and_confusion("test_part.csv",models_list,'libsvm')[0])
		test_set_accuracy.append(get_accuracy_and_confusion(test_path,models_list,'libsvm')[0])
		print validation_accuracy
		print test_set_accuracy

def get_accuracy_and_confusion(test_path,models_list,method):
	print("calculating accuracy...")
	matrix = np.zeros((10,10)).tolist()
	correctCount=0
	totalCount=0
	all_digits = np.arange(10)
	all_combinations = list(itertools.combinations(all_digits,2))
	with open(test_path,'r') as file:
		if method=='cvxopt':
			for line in file:
				prediction_counter = np.zeros(10)
				confidence_counter = np.zeros(10)
				x_temp = []
				line = line.split('\n')[0]
				line_arr = line.split(',')
				x_part = line_arr[:-1]
				y_part = line_arr[-1] 
				for i in x_part:
					x_temp.append(float(i)/255.0)
				for i in range(45):
					tup = models_list[i]
					pred = cvx_predict(tup,x_temp,all_combinations[i][0],all_combinations[i][1])
					prediction_counter[pred[0]]+=1
					confidence_counter[pred[0]]+=pred[1]
				indices = np.where(prediction_counter==max(prediction_counter))[0]
				final_pred = list(confidence_counter).index(max(confidence_counter[indices]))
				# final_pred = prediction_counter.index()
				# print(final_pred,y_part)
				matrix[int(final_pred)][int(y_part)] +=1
				if final_pred==int(y_part):
					correctCount+=1
				totalCount+=1
				sys.stdout.write("live accuracy= %.4f%% ,line no. = %d \r"%((correctCount*100./totalCount), totalCount))
				sys.stdout.flush()
		else:
			for line in file:
				prediction_counter = np.zeros(10)
				confidence_counter = np.zeros(10)
				x_temp = []
				line = line.split('\n')[0]
				line_arr = line.split(',')
				x_part = line_arr[:-1]
				y_part = line_arr[-1] 
				for i in x_part:
					x_temp.append(float(i)/255.0)
				for i in range(45):
					model = models_list[i]
					pred = libsvm_predict(model,x_temp,all_combinations[i][0],all_combinations[i][1])
					prediction_counter[pred[0]]+=1
					confidence_counter[pred[0]]+=pred[1]
				indices = np.where(prediction_counter==max(prediction_counter))[0]
				final_pred = list(confidence_counter).index(max(confidence_counter[indices]))
				# final_pred = prediction_counter.index()
				# print(final_pred,y_part)
				matrix[int(final_pred)][int(y_part)] +=1
				if final_pred==int(y_part):
					correctCount+=1
				totalCount+=1
				sys.stdout.write("live accuracy= %.4f%% ,line no. = %d \r"%((correctCount*100./totalCount), totalCount))
				sys.stdout.flush()
			# print(totalCount)
	return (correctCount*100./totalCount), matrix


# validation_method(train_path,test_path)
# models_list = generateAllClassifiers(train_path,'libsvm',gamma,C)
# res = get_accuracy_and_confusion(test_path,models_list,'libsvm')
# print("accuracy = %s %%"%res[0])
# print(res[1])
# ex = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941176470588235, 0.7254901960784313, 0.6235294117647059, 0.592156862745098, 0.23529411764705882, 0.1411764705882353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8705882352941177, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.9450980392156862, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.7764705882352941, 0.6666666666666666, 0.20392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2627450980392157, 0.4470588235294118, 0.2823529411764706, 0.4470588235294118, 0.6392156862745098, 0.8901960784313725, 0.996078431372549, 0.8823529411764706, 0.996078431372549, 0.996078431372549, 0.996078431372549, 0.9803921568627451, 0.8980392156862745, 0.996078431372549, 0.996078431372549, 0.5490196078431373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666666666666667, 0.25882352941176473, 0.054901960784313725, 0.2627450980392157, 0.2627450980392157, 0.2627450980392157, 0.23137254901960785, 0.08235294117647059, 0.9254901960784314, 0.996078431372549, 0.41568627450980394, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3254901960784314, 0.9921568627450981, 0.8196078431372549, 0.07058823529411765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08627450980392157, 0.9137254901960784, 1.0, 0.3254901960784314, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5058823529411764, 0.996078431372549, 0.9333333333333333, 0.17254901960784313, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137254901960785, 0.9764705882352941, 0.996078431372549, 0.24313725490196078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5215686274509804, 0.996078431372549, 0.7333333333333333, 0.0196078431372549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03529411764705882, 0.803921568627451, 0.9725490196078431, 0.22745098039215686, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411764705882355, 0.996078431372549, 0.7137254901960784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29411764705882354, 0.984313725490196, 0.9411764705882353, 0.2235294117647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450980392156863, 0.8666666666666667, 0.996078431372549, 0.6509803921568628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764705882352941, 0.796078431372549, 0.996078431372549, 0.8588235294117647, 0.13725490196078433, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901960784313725, 0.996078431372549, 0.996078431372549, 0.30196078431372547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156862745098039, 0.8784313725490196, 0.996078431372549, 0.45098039215686275, 0.00392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5215686274509804, 0.996078431372549, 0.996078431372549, 0.20392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23921568627450981, 0.9490196078431372, 0.996078431372549, 0.996078431372549, 0.20392156862745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098039215686, 0.996078431372549, 0.996078431372549, 0.8588235294117647, 0.1568627450980392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098039215686, 0.996078431372549, 0.8117647058823529, 0.07058823529411765, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# res = []
# for model in models_list:
# 	(support_vectors,y_i,alpha_i,b) = model
# 	res.append(vector_prediction_actual(support_vectors,ex,alpha_i,y_i,b))
# print(res)
if __name__ == "__main__":
	train_data_path = str(sys.argv[1])
	test_data_path = str(sys.argv[2])
	part_num = str(sys.argv[3])
	if part_num=="a":
		# models_list = generateAllClassifiers(train_data_path,'cvxopt',gamma,C)
		with open('models_cvxopt.pickle','rb') as infile:
			models_list =  pickle.load(infile)
		print("getting test set accuracy...")
		res = get_accuracy_and_confusion(test_data_path,models_list,'cvxopt')
		print("test set accuracy = %s %%"%res[0])
		print("getting train set accuracy...")
		res2 = get_accuracy_and_confusion(train_data_path,models_list,'cvxopt')
		print("train set accuracy = %s %%"%res2[0])

	elif part_num=="b":
		models_list = generateAllClassifiers(train_data_path,'libsvm',gamma,C)
		print("getting test set accuracy...")
		res = get_accuracy_and_confusion(test_data_path,models_list,'libsvm')
		print("test set accuracy = %s %%"%res[0])
		print("getting train set accuracy...")
		res2 = get_accuracy_and_confusion(train_data_path,models_list,'libsvm')
		print("train set accuracy = %s %%"%res2[0])
	elif part_num=="c":
		# models_list = generateAllClassifiers(train_data_path,'cvxopt',gamma,C)
		with open('models_cvxopt.pickle','rb') as infile:
			models_list =  pickle.load(infile)
		print("getting confusion matrix...")
		res = get_accuracy_and_confusion(test_data_path,models_list,'cvxopt')
		print(res[1])

	elif part_num=="d":
		print('not implemented.')