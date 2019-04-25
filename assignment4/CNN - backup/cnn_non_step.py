import _pickle as pickle
import cv2
import numpy as np
import pandas as pd
import random
import gc
import itertools
from sklearn.metrics import f1_score
import os
import time
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


#500 images per folder so aroung 30 folders per step rough number
batch_sizewa=50
epochs = 4
total_X_size = 50000
num_train = 46000
train_data_path = '/home/gulshan98125/train_dataset'
all_folders = sorted(os.listdir(train_data_path))
random.shuffle(all_folders)

model = models.Sequential()
model.add( layers.Conv2D(32, (3,3),strides=(2, 2),activation='relu',input_shape=(157, 160,5)) )
model.add( layers.MaxPooling2D((2,2),strides=(2, 2)) )

model.add( layers.Conv2D(64, (3,3),strides=(2, 2),activation='relu') )
model.add( layers.MaxPooling2D((2,2),strides=(2, 2)) )
model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.001),metrics=['acc'])


X = np.zeros((total_X_size,157,160,5))
Y = np.zeros(total_X_size)
counter = 0
for folder in all_folders:
	print(counter)
	if counter>=total_X_size:
		break
	inner_path = train_data_path+'/'+folder
	# print(inner_path, 'data_size='+str(counter))
	image_names = sorted(os.listdir(inner_path))[:-1]
	reward_arr = np.concatenate( ([[0]],np.array(pd.read_csv(inner_path+'/rew.csv',header=None))) , axis=0)
	ones_location = np.where(reward_arr.flatten()==1.0)[0]
	# print(ones_location)

	# appending frames before 1.0 reward
	for index in ones_location:
		flat_img_arr = np.array([[cv2.imread(inner_path + '/' + image_names[j], 0)[31:188]] for j in range(index-7,index,1)])
		l = range(0,6,1)
		# for comb in itertools.combinations(l,4):
		for comb in random.sample(list(itertools.combinations(l,4)),3): #pick randomly 3 images
			if counter>=total_X_size:
				break
			ind = list(comb)
			ind.append(6)
			useful_img_arr = flat_img_arr[ind] # last frame is always present
			o = np.concatenate(tuple(useful_img_arr),axis=0)
			o = o.transpose(1,-1,0)
			# print(x_pca,x_pca.shape)
			X[counter] = o
			Y[counter] = 1.0
			counter+=1

	zeros_location = random.sample(range(len(image_names)),4*len(ones_location))
	
	count_zeros = 0
	for index in zeros_location:
		if index not in ones_location:
			if count_zeros>= 2*len(ones_location): # means negative frames are twice the positive frames for an episode
				break
			flat_img_arr = np.array([[cv2.imread(inner_path + '/' + image_names[j], 0)[31:188]] for j in range(index-7,index,1)])
			l = range(0,6,1)
			# for comb in itertools.combinations(l,4):
			for comb in random.sample(list(itertools.combinations(l,4)),3):
				if counter>=total_X_size:
					break
				ind = list(comb)
				ind.append(6)
				useful_img_arr = flat_img_arr[ind]
				o = np.concatenate(tuple(useful_img_arr),axis=0)
# 				o = useful_img_arr[0]
# 				for img in useful_img_arr[1:]:
# 					o = np.concatenate((o,img),axis=0)
				o = o.transpose(1,-1,0)
				X[counter] = o
				Y[counter] = 0.0
				counter+=1
			count_zeros+=1

X_train, X_val, y_train, y_val = X[:num_train], X[num_train:], Y[:num_train], Y[num_train:],

datagen = ImageDataGenerator(rescale = 1./255)
train_generator = datagen.flow(X_train, y_train, batch_size = batch_sizewa)
val_generator  = datagen.flow(X_val, y_val, batch_size = batch_sizewa)

model.fit_generator(train_generator, steps_per_epoch = len(X_train)//batch_sizewa, epochs = epochs, validation_data = val_generator, validation_steps=len(X_val)//batch_sizewa)

pred_temp = model.predict(X_val)
pred = np.array([x[0]>0.5 for x in pred_temp]).astype(int)
print(f1_score(y_val, pred, average=None))
del X
del X_train
del X_val
gc.collect()
print("here")
outfile = open('train_cnn_model.pkl','wb')
pickle.dump(model,outfile,-1)
outfile.close()