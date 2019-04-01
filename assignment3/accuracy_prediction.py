import pandas as pd
import numpy as np
from math import log
import time

global root
global numNodes
global train_set
global test_set
global val_set
global pred_acc
pred_acc = []

isReal = [1,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]  # whether the column is real or multivalued
col_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
isRealDict = {}
for i in range(len(col_names)):
	isRealDict[col_names[i]] = isReal[i]


range_of_attr = [[], [1,2], [0,1,2,3,4,5,6], [0,1,2,3], [], list(np.arange(-2,9))\
				, list(np.arange(-2,9)), list(np.arange(-2,9)), list(np.arange(-2,9)), list(np.arange(-2,9)), list(np.arange(-2,9))\
				, [], [], [], [], [], [], [], [], [], [], [], []]
range_of_attr_dict = {}
for i in range(len(col_names)):
	range_of_attr_dict[col_names[i]] = range_of_attr[i]


dataframe = pd.read_csv('cc_train.csv')
# dataframe = dataframe.drop(index=0,columns=['Unnamed: 0','Y'])
dataframe = dataframe.drop(index=0,columns=['X0'])
dataframe = dataframe.astype(int)
median_val = dataframe.median() 

class treeNode():
	def __init__(self,dataframe,finalVal):
		self.attribute = None
		self.children = None
		self.df = dataframe
		self.finalVal = finalVal
		self.function = None	#retuns true or false accoring to function

def predict(node,row_df):	#recursive prediction by updating the node
	# if node.finalVal!=None:
	# 	return node.finalVal
	if node.attribute == None:
		num_pos = node.df['Y'].astype(bool).sum(axis=0)
		num_neg = len(node.df) - num_pos
		if num_pos>=num_neg:
			return 1
		else:
			return 0
	else:
		attr = node.attribute
		children = node.children
		val = row_df[attr]
		for child in children:
			# print(child.finalVal)
			if child.function(val)==True:
				return predict(child,row_df)

def predict_df(node, df):		#gives accuracy over dataframe
	print("predicting dataframe")
	totalCount = len(df)
	# prediction = []
	correct = 0.0
	for i in range(len(df)):
		p = predict(node,df.iloc[i])
		if p==df.iloc[i]['Y']:
			correct+=1
		# prediction.append(p)
	# f_score = f1_score(np.array(df['Y']), prediction, average='macro')
	return correct/totalCount, correct,totalCount

def getEntropy(numPositive,numNegative):
	if numPositive==0 and numNegative==0:
		return 0.0
	else:
		pos_prob = (1.0*numPositive)/(numPositive+numNegative)
		neg_prob = 1.0-pos_prob
		if numPositive==0:
			return -((neg_prob)*log(neg_prob,2))
		elif numNegative==0:
			return -((pos_prob)*log(pos_prob,2))
		else:
			return -((pos_prob)*log(pos_prob,2)+(neg_prob)*log(neg_prob,2))


# given a DF and a feature column it gives all the possible gain
def gain(currentDF, feature, method): #method = 'global' median, 'local' median, global means preprocessing columns
	parent_num_pos = currentDF['Y'].astype(bool).sum(axis=0)
	parent_num_neg = len(currentDF) - parent_num_pos
	parent_entropy = getEntropy(parent_num_pos,parent_num_neg)

	if isRealDict[feature] == 1:
		if method=='global':
			median = median_val[feature]
		else:
			median = currentDF.median()[feature]

		df_child1_Y = currentDF['Y'].loc[ currentDF[feature]<median ]
		df_child2_Y = currentDF['Y'].loc[ currentDF[feature]>=median ]
		
		child1_num_pos = df_child1_Y.astype(bool).sum(axis=0)
		child1_num_neg = len(df_child1_Y)- child1_num_pos
		
		child2_num_pos = df_child2_Y.astype(bool).sum(axis=0)
		child2_num_neg = len(df_child2_Y)- child2_num_pos
		child1_entropy = getEntropy(child1_num_pos,child1_num_neg)
		child2_entropy = getEntropy(child2_num_pos,child2_num_neg)

		gain_val = parent_entropy - (1.0*len(df_child1_Y)/len(currentDF))*child1_entropy - (1.0*len(df_child2_Y)/len(currentDF))*child2_entropy
		return gain_val

	else:
		attr_values = range_of_attr_dict[feature]
		df_children_arr = []
		for i in range(len(attr_values)):
			df_children_arr.append(currentDF['Y'].loc[ currentDF[feature]==attr_values[i] ])

		gain_val = parent_entropy
		for i in range(len(attr_values)):
			num_pos = df_children_arr[i].astype(bool).sum(axis=0)
			num_neg = len(df_children_arr[i])- num_pos
			# print(num_pos,num_neg)
			gain_val -= (1.0*len(df_children_arr[i])/len(currentDF))*getEntropy(num_pos,num_neg)
		return gain_val



def growTree(node,method): #given a node with its dataframe construct the tree recursively and return that node
	global numNodes
	global pred_acc
	global test_set
	global val_set
	if numNodes%5==0:
		pred = predict_df(root,val_set)
		pred_acc.append(pred[0])
		print("val prediction acc on %d nodes = %f"%(numNodes,pred[0]*100))
	numNodes+=1
	total = len(node.df)
	num_pos = node.df['Y'].astype(bool).sum(axis=0)
	num_neg = total-num_pos
	# print("DF",node.df)
	if num_pos ==0:
		node.finalVal=0
		return
	elif num_neg==0:
		node.finalVal=1
		return
	else:
		#-------------finding best split attribute-------------
		gain_list = []
		for i in col_names:
			gain_list.append(gain(node.df,i,method))


		best_attr = col_names[gain_list.index(max(gain_list))]

		if max(gain_list)==0:
			if num_pos>=num_neg:
				node.finalVal=1
				return
			else:
				node.finalVal=0
				return
		node.attribute = best_attr          # setting attribute
		#------------------------------------------------------

		if isRealDict[best_attr] == 1:
			if method=='global':
				median = median_val[best_attr]
			else:
				median = node.df.median()[best_attr]

			df_child1 = node.df.loc[ node.df[best_attr]<median ] #dataframe for child1
			df_child2 = node.df.loc[ node.df[best_attr]>=median ] #dataframe for child2
			childNode1 = treeNode(df_child1,None)
			def generate1(v): return lambda y: y<v
			def generate2(v): return lambda y: y>=v

			childNode1.function = generate1(median)
			childNode2 = treeNode(df_child2,None)

			childNode2.function = generate2(median)


			children_arr = [childNode1,childNode2]
			node.children = children_arr
			node.df = None
			growTree(childNode1,method)
			growTree(childNode2,method)
		else:
			attr_values = range_of_attr_dict[best_attr]
			# df_children_arr = []

			# for i in range(len(attr_values)):	#calculate child's dataframe
			# 	df_children_arr.append(node.df.loc[ node.df[best_attr]==attr_values[i] ])
			
			def generate(v): return lambda y: y == v

			children_arr = [None]*len(attr_values)
			for j in range(len(attr_values)):	#create child node
				children_arr[j] = treeNode(node.df.loc[ node.df[best_attr]==attr_values[j] ],None)
				# f = lambda y: y==attr_values[j]
				children_arr[j].function=generate(attr_values[j])

			node.children = children_arr
			node.df = None
			for k in range(len(children_arr)):	# grow children trees
				growTree(children_arr[k],method)

numNodes = 0
train_set=dataframe
test_set = pd.read_csv('cc_test.csv')
test_set = test_set.drop(index=0,columns=['X0'])
test_set = test_set.astype(int)

val_set = pd.read_csv('cc_val.csv')
val_set = val_set.drop(index=0,columns=['X0'])
val_set = val_set.astype(int)

root = treeNode(dataframe,None)
# print("growing...")
# start_time = time.time()
# growTree(root,'local')
# print("grow time = %f"%(time.time()-start_time))
# outfile = open('tree.pkl','wb')
# pickle.dump(root,outfile,-1)
# for i in range(20):
# 	print predict(root,dataframe.iloc[i])

# df = dataframe.iloc[[1692,3165,19106]]
# print(gain(df,'X13','local'))
