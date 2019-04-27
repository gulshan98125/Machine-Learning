import _pickle as pickle
infile = open('cnn_model.pkl','rb')
model = pickle.load(infile)
infile.close()
import cv2
import numpy as np
import pandas as pd
import random
import gc
import itertools
from sklearn.metrics import f1_score
import os
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


test_data_path = '/home/gulshan98125/test_dataset/test_dataset'
all_folders = sorted(os.listdir(test_data_path))
X_test = np.zeros((len(all_folders),157,160,5))

counter = 0
for folder in all_folders:
	inner_path = test_data_path+'/'+folder
	image_names = sorted(os.listdir(inner_path))
	img_arr = np.array([[cv2.imread(inner_path + '/' + img_name, 0)[31:188]] for img_name in image_names])
	o = np.concatenate(tuple(img_arr),axis=0)
	o = o.transpose(1,-1,0)
	o = o/255.
	X_test[counter] = o
	counter+=1
y_pred_temp = model.predict(X_test)
Y_PRED = np.array([x[0]>0.5 for x in y_pred_temp]).astype(int)
del X_test
Y_PRED = Y_PRED.astype(int)
df = pd.DataFrame(data=Y_PRED)
df.to_csv('out.csv')
