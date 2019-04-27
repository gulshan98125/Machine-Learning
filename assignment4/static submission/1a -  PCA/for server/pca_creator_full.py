import numpy as np
import pandas as pd
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import _pickle as pickle

train_data_path = '/home/gulshan98125/train_dataset'
pca = PCA(n_components=50)
scaler = StandardScaler()
num_images = 0
outer_i = 0
#counting number of images in first 50 episodes
for folder in sorted(os.listdir(train_data_path)):
	if outer_i==50:
		break
	inner_path = train_data_path+'/'+folder
	num_images+= len(os.listdir(inner_path))-1
	outer_i+=1

inner_i=0
data = np.zeros((num_images,179*160))
for folder in sorted(os.listdir(train_data_path)):
	if inner_i==num_images:
		break
	inner_path = train_data_path+'/'+folder
	print(inner_path)
	for img_name in sorted(os.listdir(inner_path))[:-1]:
		if inner_i==num_images:
			break
		img_path = train_data_path+'/'+folder+'/'+img_name
		img = cv2.imread(img_path, 0)[31:]
		flat_img = img.flatten()/255.
		data[inner_i] = flat_img
		inner_i+=1

start = time.time()
scaler.fit(data)
data = scaler.transform(data)
print("scaler time:",time.time()-start)

start = time.time()
pca.fit(data)
print("pca calculation time:",time.time()-start)
outfile = open('scaler_and_pca_model.pkl','wb')
pickle.dump((scaler,pca),outfile,-1)
outfile.close()
del pca
del data