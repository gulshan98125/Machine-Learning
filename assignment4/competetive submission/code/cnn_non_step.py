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
class CNN():
	def __init__(self,train_path, img_height,img_width,batch_size=50,epochs=4,learning_rate = 0.001,train_data_size=50000, fraction_train = 0.9):
		self.batch_size = batch_size
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.train_data_size = train_data_size
		self.fraction_test = 1.0 - fraction_train
		self.train_path = train_path
		self.img_height = img_height
		self.img_width = img_width

		model = models.Sequential()
		model.add( layers.Conv2D(64, (6,6),strides=(2, 2),activation='relu',input_shape=(self.img_height, self.img_width,5)) )
		model.add( layers.MaxPooling2D((2,2),strides=(2, 2)) )
		model.add( layers.Conv2D(64, (6,6),strides=(2, 2),activation='relu') )
		model.add( layers.MaxPooling2D((2,2),strides=(2, 2)) )
		model.add( layers.Conv2D(64, (6,6),strides=(2, 2),activation='relu') )
		model.add( layers.MaxPooling2D((2,2),strides=(2, 2)) )
		model.add(layers.Flatten())
		model.add(layers.Dense(1024, activation='relu'))
		model.add(layers.Dense(2048, activation='relu'))
		model.add(layers.Dense(1,activation='sigmoid'))
		model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=learning_rate),metrics=['acc'])
		self.model = model

	def split_data(self,X, Y, fraction_test):
		ratio = int(X.shape[0]*fraction_test) #should be int
		indexes = range(len(X))
		random.shuffle(X)
		train_ind = indexes[ratio:]
		test_ind = indexes[:ratio]

		X_train, Y_train = X[train_ind], Y[train_ind]
		X_test, Y_test = X[test_ind], Y[test_ind]
		return X_train, X_test, Y_train, Y_test

	def fit(self):
		all_folders = os.listdir(self.train_path)
		random.shuffle(all_folders)
		X = np.zeros((self.train_data_size,self.img_height,self.img_width,5))
		Y = np.zeros(self.train_data_size)
		counter = 0
		print("generating dataset...")
		print("total size:",self.train_data_size)
		for folder in all_folders:
			print(counter)
			if counter>=self.train_data_size:
				break
			inner_path = self.train_path+'/'+folder
			image_names = sorted(os.listdir(inner_path))[:-1]
			reward_arr = np.concatenate( ([[0]],np.array(pd.read_csv(inner_path+'/rew.csv',header=None))) , axis=0)
			ones_location = np.where(reward_arr.flatten()==1.0)[0]

			# appending frames before 1.0 reward
			for index in ones_location:
				flat_img_arr = np.array([[cv2.imread(inner_path + '/' + image_names[j], 0)[31:188]] for j in range(index-7,index,1)])
				l = range(0,6,1)
				for comb in random.sample(list(itertools.combinations(l,4)),3): #pick randomly 3 images
					if counter>=self.train_data_size:
						break
					ind = list(comb)
					ind.append(6)
					useful_img_arr = flat_img_arr[ind] # last frame is always present
					o = np.concatenate(tuple(useful_img_arr),axis=0)
					o = o.transpose(1,-1,0)
					X[counter] = o
					Y[counter] = 1.0
					counter+=1

			zeros_location = random.sample(range(len(image_names)),min(len(image_names)/2,5*len(ones_location)))
			
			count_zeros = 0
			for index in zeros_location:
				if index not in ones_location:
					if count_zeros>= 2*len(ones_location): # means negative frames are twice the positive frames for an episode
						break
					flat_img_arr = np.array([[cv2.imread(inner_path + '/' + image_names[j], 0)[31:188]] for j in range(index-7,index,1)])
					l = range(0,6,1)
					for comb in random.sample(list(itertools.combinations(l,4)),3):
						if counter>=self.train_data_size:
							break
						ind = list(comb)
						ind.append(6)
						useful_img_arr = flat_img_arr[ind]
						o = np.concatenate(tuple(useful_img_arr),axis=0)
						o = o.transpose(1,-1,0)
						X[counter] = o
						Y[counter] = 0.0
						counter+=1
					count_zeros+=1

		# X_train, X_val, y_train, y_val = self.split_data(X,Y,self.fraction_test)
		num_train = int(X.shape[0]*(1-self.fraction_test))
		X_train, X_val, y_train, y_val = X[:num_train], X[num_train:], Y[:num_train], Y[num_train:]
		del X
		datagen = ImageDataGenerator(rescale = 1./255)
		train_generator = datagen.flow(X_train, y_train, batch_size = self.batch_size)
		val_generator  = datagen.flow(X_val, y_val, batch_size = self.batch_size)
		print("training model...")
		self.model.fit_generator(train_generator, steps_per_epoch = len(X_train)//self.batch_size, epochs = self.epochs, validation_data = val_generator, validation_steps=len(X_val)//self.batch_size)
		pred_temp = self.model.predict(X_val)
		pred = np.array([x[0]>0.5 for x in pred_temp]).astype(int)
		print("f score: ",f1_score(y_val, pred, average=None))
		del train_generator
		del val_generator
		del X_train
		del X_val
		gc.collect()


	def predict(self,X):
		return self.model.predict(X)

	def save(self,file_name):
		outfile = open(str(file_name),'wb')
		pickle.dump(self.model,outfile,-1)
		outfile.close()


# batch_sizewa=50
# epochs = 4
# self.train_data_size = 50000
# num_train = 46000
train_data_path = '/home/gulshan98125/train_dataset'

m = CNN(train_path = train_data_path,train_data_size = 60000, img_height=157, img_width=160)
m.fit()
m.save('cnn_model.pkl')