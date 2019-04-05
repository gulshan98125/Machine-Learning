import pandas as pd
import sys
if __name__ == "__main__":
	train_path = str(sys.argv[1])
	test_path = str(sys.argv[2])
	one_hot_train_path = str(sys.argv[3])
	one_hot_test_path = str(sys.argv[4])
	# train_path = 'poker_train.csv'
	# test_path = 'poker_test.csv'
	# one_hot_train_path = 'train_output.csv'
	# one_hot_test_path = 'test_output.csv'

	train = pd.read_csv(train_path,header=None)
	train.columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','Y']
	test = pd.read_csv(test_path,header=None)
	test.columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','Y']

	df2 = pd.DataFrame([\
	[1, 1,  1, 1,  1, 1,  1,  1,  1,  1,  0],\
	[2, 2,  2, 2,  2, 2,  2,  2,  2,  2,  0],\
	[3, 3,  3, 3,  3, 3,  3,  3,  3,  3,  0],\
	[4, 4,  4, 4,  4, 4,  4,  4,  4,  4,  0],\
	[4, 5,  1, 5,  1, 5,  1,  5,  1,  5,  0],\
	[4, 6,  1, 6,  1, 6,  1,  6,  1,  6,  0],\
	[4, 7,  1, 7,  1, 7,  1,  7,  1,  7,  0],\
	[4, 8,  1, 8,  1, 8,  1,  8,  1,  8,  0],\
	[4, 9,  1, 9,  1, 9,  1,  9,  1,  9,  0],\
	[4, 10, 1, 10, 1, 10, 1, 10,  1,  10, 0],\
	[4, 11, 1, 11, 1, 11, 1, 11,  1,  11, 0],\
	[4, 12, 1, 12, 1, 12, 1, 12,  1,  12, 0],\
	[4, 13, 1, 13, 1, 13, 1, 13,  1,  13, 0],\
	], columns=train.columns)

	train = df2.append(train)
	test = df2.append(test)
	dictwa = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h',9:'i',10:'j',11:'k',12:'l',13:'m',0:'n'}
	sets = [train,test]
	for df in sets:
		for i in range(1,11):
			col = 'X'+str(i)
			df[col] = df[col].map(dictwa)

	test = pd.get_dummies(test)
	train = pd.get_dummies(train)

	test = test[13:]
	train = train[13:]
	print("saving...")
	train.to_csv(one_hot_train_path)
	test.to_csv(one_hot_test_path)

	"""final columns = [u'X1_a', u'X1_b', u'X1_c', u'X1_d', u'X2_a', u'X2_b', u'X2_c', u'X2_d',
       u'X2_e', u'X2_f', u'X2_g', u'X2_h', u'X2_i', u'X2_j', u'X2_k', u'X2_l',
       u'X2_m', u'X3_a', u'X3_b', u'X3_c', u'X3_d', u'X4_a', u'X4_b', u'X4_c',
       u'X4_d', u'X4_e', u'X4_f', u'X4_g', u'X4_h', u'X4_i', u'X4_j', u'X4_k',
       u'X4_l', u'X4_m', u'X5_a', u'X5_b', u'X5_c', u'X5_d', u'X6_a', u'X6_b',
       u'X6_c', u'X6_d', u'X6_e', u'X6_f', u'X6_g', u'X6_h', u'X6_i', u'X6_j',
       u'X6_k', u'X6_l', u'X6_m', u'X7_a', u'X7_b', u'X7_c', u'X7_d', u'X8_a',
       u'X8_b', u'X8_c', u'X8_d', u'X8_e', u'X8_f', u'X8_g', u'X8_h', u'X8_i',
       u'X8_j', u'X8_k', u'X8_l', u'X8_m', u'X9_a', u'X9_b', u'X9_c', u'X9_d',
       u'X10_a', u'X10_b', u'X10_c', u'X10_d', u'X10_e', u'X10_f', u'X10_g',
       u'X10_h', u'X10_i', u'X10_j', u'X10_k', u'X10_l', u'X10_m', u'Y_a',
       u'Y_b', u'Y_c', u'Y_d', u'Y_e', u'Y_f', u'Y_g', u'Y_h', u'Y_i', u'Y_n'],
      dtype='object') """