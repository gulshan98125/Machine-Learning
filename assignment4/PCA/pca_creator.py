# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import cPickle as pickle
# x_pca=pca.transform(scaled_data)
# print x_pca

train_data_path = '../data/train'
pca=IncrementalPCA(n_components=50)
outer_i = 0
inner_i=0
for folder in os.listdir(train_data_path):
	inner_path = train_data_path+'/'+folder
	# reward_arr = np.concatenate( ([[0]],np.array(pd.read_csv(inner_path+'/rew.csv'))) , axis=0)
	temp_img_arr = np.ndarray((2000,193*160))
	for img_name in sorted(os.listdir(inner_path))[:-1]:
		img_path = train_data_path+'/'+folder+'/'+img_name
		img = cv2.imread(img_path, 0)[17:]
		flat_img = img.flatten()/255.
		# temp_img_arr[inner_i] = np.concatenate((flat_img, reward_arr[inner_i]))
		temp_img_arr[inner_i] = flat_img
		print(img_path)
		inner_i+=1
		if inner_i==1999 :
			inner_i=0
			pca.partial_fit(temp_img_arr)
	outer_i+=1
	if outer_i==50:
		if temp_img_arr.shape[0]>=50:
			pca.partial_fit(temp_img_arr)
		break
outfile = open('pca_model.pkl','wb')
pickle.dump(pca,outfile,-1)
outfile.close()
del pca
del temp_img_arr