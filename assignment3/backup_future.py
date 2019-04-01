import pandas as pd
import numpy as np
from math import log
import time
import copy
from sklearn.metrics import f1_score

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


class treeNode():
	def __init__(self,dataframe,finalVal):
		self.attribute = None
		self.children = None
		self.df = dataframe
		self.finalVal = finalVal
		self.function = None	#retuns true or false accoring to function
		self.numPos = None
		self.numNeg = None
		self.indexes_array = []

	# def function(self,v):



def growTree(node,method): #given a node with its dataframe construct the tree recursively and return that node
	total = len(node.df)
	num_pos = node.df['Y'].astype(bool).sum(axis=0)
	num_neg = total-num_pos
	node.numPos = num_pos
	node.numNeg = num_neg
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
			# print(node.df)
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



def predict(node,row_df):	#recursive prediction by updating the node
	# if node.finalVal!=None:
	# 	return node.finalVal
	if node.children == []:
		num_pos = node.df['Y'].astype(bool).sum(axis=0)
		num_neg = len(node.df) - num_neg
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

root = treeNode(dataframe,None)
print("growing...")
start_time = time.time()
growTree(root,'local')
print("runtime = %f"%(time.time()-start_time))
# outfile = open('tree.pkl','wb')
# pickle.dump(root,outfile,-1)
# for i in range(20):
# 	print predict(root,dataframe.iloc[i])

# df = dataframe.iloc[[1692,3165,19106]]
# print(gain(df,'X13','local'))

def predict_df(node, df):		#gives accuracy over dataframe
	print("predicting dataframe")
	totalCount = len(df)
	prediction = []
	correct = 0.0
	for i in range(len(df)):
		p = predict(node,df.iloc[i])
		if p==df.iloc[i]['Y']:
			correct+=1
		prediction.append(p)
	f_score = f1_score(np.array(df['Y']), prediction, average='macro')
	return correct/totalCount,f_score, correct,totalCount


def update_indexes(node, row_df, index):
	if node.finalVal != None:
		return
	node.indexes_array.append(index)
	attr = node.attribute
	children = node.children
	val = row_df[attr]
	for child in children:
		if child.function(val)==True:
			update_indexes(child, row_df, index)

def update_evaluation_indexes(root_node,df):	#given a dataframe, stores indexes passing through the nodes (of that df)
	for i in range(len(df)):
		update_indexes(root_node,df.iloc[i],i)


def pruneTree(node, df): #prunes tree in 1 go for all the parent of leaf nodes
	nonLeafChildList = []
	for child in node.children:
		if child.finalVal == None:
			nonLeafChildList.append(child)
	if nonLeafChildList!=[]:
		for X in nonLeafChildList:
			pruneTree(X, df)
	else:
		useful_df = df.iloc[node.indexes_array]
		num_pos = useful_df['Y'].astype(bool).sum(axis=0)
		if majority_pred_acc > pred_acc:
			node.finalVal = majority_pred_acc
			return
		else:
			return




def numNodes(node):
	if node.finalVal != None:
		return 0
	count = 1
	for i in node.children:
		count +=  numNodes(i)
	return count

df_test = pd.read_csv('cc_test.csv')
df_test = df_test.drop(index=0,columns=['X0'])
df_test = df_test.astype(int)

df_val = pd.read_csv('cc_val.csv')
df_val = df_val.drop(index=0,columns=['X0'])
df_val = df_val.astype(int)

df_train = dataframe


# def cutTree(node, num_nodes):


# def print_accuracy_with_numNodes(node,df):
# 	print("updating indexes")
# 	update_evaluation_indexes(node, df)
# 	for i in range(0,len(df),100):

