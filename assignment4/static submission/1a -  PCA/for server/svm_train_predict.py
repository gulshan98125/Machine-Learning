import _pickle as pickle
import cv2
import os
import numpy as np
import pandas as pd
import random
import itertools
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
gamma = 0.05
C = 0.2

infile = open('scaler_and_pca_model.pkl','rb')
scaler,pca = pickle.load(infile)
infile.close()

infile = open('train_XY.pkl','rb')
(X,Y) = pickle.load(infile)
infile.close()

scaler2 = StandardScaler()
scaler2.fit(X)
X_n = scaler2.transform(X)

svm = SVC(kernel='rbf',C=C, gamma =gamma)
svm.fit(X_n[:15000],Y[:15000])
y_pred = svm.predict(X_n[15000:])
y_pred2 = svm.predict(X_n[:15000])
print("f1 score train: ",f1_score(Y[:15000], y_pred2, average=None))
print("f1 score validation: ",f1_score(Y[15000:], y_pred, average=None))
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