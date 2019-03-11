import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

train_path = '../data/train.csv'
test_path = '../data/test.csv'
C = 1.0

X = []
Y = []
with open(train_path,'r') as train_file:
	for line in train_file:
		x_temp = []
		line = line.split('\n')[0]
		line_arr = line.split(',')
		x_part = line_arr[:-1]
		y_part = line_arr[-1]
		if int(y_part)==5:
			Y.append(-1.0)
			for i in x_part:
				x_temp.append(float(i))
			X.append(x_temp)
		elif int(y_part)==6:
			Y.append(1.0)
			for i in x_part:
				x_temp.append(float(i))
			X.append(x_temp)
X = np.array(X)
Y = np.array(Y)
