import cPickle as pickle
import cv2
import numpy as np
import pandas as pd
import svmutil as SVM
import random
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
import os
SVM.svm_model.predict = lambda self, x: SVM.svm_predict([0], [x], self, "-q")[0][0]
gamma = 0.05
C = 1.0
total_X_size = 20000

infile = open('pca_model_grayscale.pkl','rb')
pca = pickle.load(infile)
infile.close()


X = np.zeros((total_X_size,250))
Y = np.zeros(total_X_size)
counter = 0
train_data_path = '../data/train'
all_folders = os.listdir(train_data_path)
random.shuffle(all_folders)
for folder in all_folders:
	if counter>=total_X_size:
		break
	inner_path = train_data_path+'/'+folder
	print(inner_path, 'data_size='+str(counter))
	image_names = sorted(os.listdir(inner_path)[:-1])
	reward_arr = np.concatenate( (np.array(pd.read_csv(inner_path+'/rew.csv',header=None)),[[0]]) , axis=0)
	ones_location = np.where(reward_arr.flatten()==1.0)[0]
	# print(ones_location)

	# appending frames before 1.0 reward
	for index in ones_location:
		flat_img_arr = np.array([cv2.imread(inner_path + '/' + image_names[j], 0)[17:].flatten()/255. for j in range(index-7,index,1)])
		l = range(0,6,1)
		# for comb in itertools.combinations(l,4):
		for comb in random.sample(list(itertools.combinations(l,4)),2): #pick randomly 3 images
			if counter>=total_X_size:
				break
			ind = list(comb)
			ind.append(6)
			useful_img_arr = flat_img_arr[ind] # last frame is always present

			#useful_img_arr has 5 images
			x_pca=pca.transform(useful_img_arr)
			# print(x_pca,x_pca.shape)
			X[counter] = x_pca.flatten()
			Y[counter] = 1.0
			counter+=1

	zeros_location = random.sample(range(len(image_names)),4*len(ones_location))
	
	count_zeros = 0
	for index in zeros_location:
		if index not in ones_location:
			if count_zeros>= 3*len(ones_location): # means negative frames are twice the positive frames for an episode
				break
			flat_img_arr = np.array([cv2.imread(inner_path + '/' + image_names[j], 0)[17:].flatten()/255. for j in range(index-7,index,1)])
			l = range(0,6,1)
			# for comb in itertools.combinations(l,4):
			for comb in random.sample(list(itertools.combinations(l,4)),2):
				if counter>=total_X_size:
					break
				ind = list(comb)
				ind.append(6)
				useful_img_arr = flat_img_arr[ind]
				x_pca=pca.transform(useful_img_arr)
				X[counter] = x_pca.flatten()
				Y[counter] = -1.0
				counter+=1
			count_zeros+=1
# model = SVM.svm_train( Y, X, "-s 0 -c "+str(C)+" -t 2 -e 0.1 -g "+str(gamma)) #first element is fake
print("here")
outfile = open('train_XY_test.pkl','wb')
pickle.dump((X,Y),outfile,-1)
outfile.close()

# y_true = Y
# y_pred = []
# for i in range(len(X)):
# 	y_pred.append(model.predict(X[i]))
# yy_true = np.array(y_true)
# yy_true[yy_true==0.0] = -1.0
# yy_pred = np.array(y_pred)
# print("test accuracy = %f percent" % ((sum(yy_true==yy_pred)*100.0)/len(yy_true)) )
# print(f1_score(yy_true, yy_pred, average=None))


# full_img_arr = np.zeros((30000,250))
# rewards_df = pd.read_csv('../data/validation/rewards.csv', header=None).drop(columns=[0])
# validation_rewards = np.array(rewards_df).flatten()
# y_true = validation_rewards
# y_pred = []
# for folder in os.listdir('data/validation')[:-1]:
# 	image_names = sorted(os.listdir('data/validation/'+folder))
# 	flat_img_arr = np.array([cv2.imread('data/validation/'+folder+ '/' + img_name, 0)[17:].flatten()/255. for img_name in image_names])
# 	x_pca=pca.transform(flat_img_arr)
# 	pred = model.predict(x_pca.flatten())
# 	y_pred.append(pred)
# yy_true = np.array(y_true)
# yy_true[yy_true==0.0] = -1.0
# yy_pred = np.array(y_pred)
# print((sum(yy_true==yy_pred)*100.0)/len(yy_true), f1_score(yy_true, yy_pred, average=None))
# print(fbeta_score(yy_true, yy_pred,average=None,beta=2))
# print(fbeta_score(yy_true, yy_pred,average=None,beta=0.5))

# for folder in os.listdir(train_data_path):
# 	inner_path = train_data_path+'/'+folder
# 	image_names = os.listdir(inner_path)[:-1]
# 	reward_arr = np.concatenate( ([[0]],np.array(pd.read_csv(inner_path+'/rew.csv',header=None))) , axis=0).flatten()
# 	y_true = []
# 	y_pred = []
# 	print("predicting")
# 	for i in range(0,len(reward_arr)-7,1):
# 		flat_img_arr = np.array([cv2.imread(inner_path + '/' + image_names[j], 0)[17:].flatten()/255. for j in range(i,i+7,1)])

# 		l = range(0,6,1)
# 		image_indexes = list(list(itertools.combinations(l,4))[np.random.randint(0,15)]) + [6] #random 5 images out of 7, last is included
# 		useful_img_arr = flat_img_arr[image_indexes]
# 		x_pca=pca.transform(useful_img_arr)
# 		pred = model.predict(x_pca.flatten())
# 		y_true.append(reward_arr[i+7])
# 		y_pred.append(pred)
# 	yy_true = np.array(y_true)
# 	yy_true[yy_true==0.0] = -1.0
# 	yy_pred = np.array(y_pred)
# 	print((sum(yy_true==yy_pred)*100.0)/len(yy_true), f1_score(yy_true, yy_pred, average=None))
# 	print(fbeta_score(yy_true, yy_pred,average=None,beta=2))
# 	print(fbeta_score(yy_true, yy_pred,average=None,beta=0.5))

#model = svmtrain(trainLabels, trainFeatures, '-h 0 -b 1 -s 0 -c 10 -w1 0.5 -w-1 0.003');