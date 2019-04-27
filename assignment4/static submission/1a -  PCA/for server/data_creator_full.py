import _pickle as pickle
import cv2
import numpy as np
import pandas as pd
import random
import itertools
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import os

total_X_size = 20000

infile = open('scaler_and_pca_model.pkl','rb')
scaler,pca = pickle.load(infile)
infile.close()


X = np.zeros((total_X_size,250))
Y = np.zeros(total_X_size)
counter = 0
train_data_path = '/home/gulshan98125/train_dataset'
all_folders = os.listdir(train_data_path)
random.shuffle(all_folders)

for folder in all_folders:
	if counter>=total_X_size:
		break
	inner_path = train_data_path+'/'+folder
	print(inner_path, 'data_size='+str(counter))
	image_names = sorted(os.listdir(inner_path))[:-1]
	reward_arr = np.concatenate( ([[0]], np.array(pd.read_csv(inner_path+'/rew.csv',header=None))) , axis=0)
	ones_location = np.where(reward_arr.flatten()==1.0)[0]

	# appending frames before 1.0 reward
	for index in ones_location:
		flat_img_arr = np.array([cv2.imread(inner_path + '/' + image_names[j], 0)[31:].flatten()/255. for j in range(index-7,index,1)])
		l = range(0,6,1)
		# for comb in itertools.combinations(l,4):
		for comb in random.sample(list(itertools.combinations(l,4)),2): #pick randomly 3 images
			if counter>=total_X_size:
				break
			ind = list(comb)
			ind.append(6)

			#useful_img_arr has 5 images
			useful_img_arr = flat_img_arr[ind] # last frame is always present
			useful_img_arr = scaler.transform(useful_img_arr)
			x_pca=pca.transform(useful_img_arr)
			# print(x_pca,x_pca.shape)
			X[counter] = x_pca.flatten()
			Y[counter] = 1.0
			counter+=1

	zeros_location = random.sample(range(len(image_names)),4*len(ones_location))
	
	count_zeros = 0
	for index in zeros_location:
		if index not in ones_location:
			if count_zeros>= 2*len(ones_location): # means negative frames are twice the positive frames for an episode
				break
			flat_img_arr = np.array([cv2.imread(inner_path + '/' + image_names[j], 0)[31:].flatten()/255. for j in range(index-7,index,1)])
			l = range(0,6,1)
			# for comb in itertools.combinations(l,4):
			for comb in random.sample(list(itertools.combinations(l,4)),2):
				if counter>=total_X_size:
					break
				ind = list(comb)
				ind.append(6)
				useful_img_arr = flat_img_arr[ind]
				useful_img_arr = scaler.transform(useful_img_arr)
				x_pca=pca.transform(useful_img_arr)
				X[counter] = x_pca.flatten()
				Y[counter] = -1.0
				counter+=1
			count_zeros+=1
print("saving train_data")
outfile = open('train_XY.pkl','wb')
pickle.dump((X,Y),outfile,-1)
outfile.close()
