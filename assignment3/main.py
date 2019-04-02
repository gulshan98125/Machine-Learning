import pandas as pd
import numpy as np
from math import log
import time
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
global root

isReal = [1,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]  # whether the column is real or multivalued
col_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
isRealDict = {}
for i in range(len(col_names)):
	isRealDict[col_names[i]] = isReal[i]


range_of_attr = [[], [1,2], [0,1,2,3,4,5,6], [0,1,2,3], [], list(np.arange(-2,10))\
				, list(np.arange(-2,10)), list(np.arange(-2,10)), list(np.arange(-2,10)), list(np.arange(-2,10)), list(np.arange(-2,10))\
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
	def __init__(self,df):
		self.attribute = None
		self.children = None
		self.df = df
		self.finalVal = None
		self.function = None	#retuns true or false accoring to function
		self.indexes_array = []

def predict(node,row_df):	#recursive prediction by updating the node
	if node.finalVal!=None:
		return node.finalVal
	else:
		attr = node.attribute
		children = node.children
		val = row_df[attr]
		for child in children:
			# print(child.finalVal)
			if child.function(val)==True:
				return predict(child,row_df)

def predict_df(node, df):		#gives accuracy over dataframe
	totalCount = len(df)
	# prediction = []
	correct = 0.0
	correct_zeros = 0.0
	correct_ones = 0.0
	for i in range(len(df)):
		p = predict(node,df.iloc[i])
		if p==df.iloc[i]['Y']:
			correct+=1
			if p==0:
				correct_zeros+=1
			else:
				correct_ones+=1
		# prediction.append(p)
	# f_score = f1_score(np.array(df['Y']), prediction, average='macro')
	return correct/totalCount, correct,totalCount, correct_zeros, correct_ones

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
		gain_val = parent_entropy
		for i in range(len(attr_values)):
			df_child = currentDF['Y'].loc[ currentDF[feature]==attr_values[i] ]
			num_pos = df_child.astype(bool).sum(axis=0)
			num_neg = len(df_child)- num_pos
			# print(num_pos,num_neg)
			gain_val -= (1.0*len(df_child)/len(currentDF))*getEntropy(num_pos,num_neg)
		return gain_val



def growTree(node,method): #given a node with its dataframe construct the tree recursively and return that node
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
			childNode1 = treeNode(df_child1)
			def generate1(v): return lambda y: y<v
			def generate2(v): return lambda y: y>=v

			childNode1.function = generate1(median)
			childNode2 = treeNode(df_child2)

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
				children_arr[j] = treeNode(node.df.loc[ node.df[best_attr]==attr_values[j] ])
				# f = lambda y: y==attr_values[j]
				children_arr[j].function=generate(attr_values[j])

			node.children = children_arr
			node.df = None
			for k in range(len(children_arr)):	# grow children trees
				growTree(children_arr[k],method)

def update_indexes_temp(node, row_df, index):
	if node.finalVal != None:
		return
	node.indexes_array.append(index)
	attr = node.attribute
	children = node.children
	val = row_df[attr]
	for child in children:
		if child.function(val)==True:
			update_indexes_temp(child, row_df, index)

def update_indexes(root_node,df):	#given a dataframe, stores indexes passing through the nodes (of that df)
	print("updating indexes")
	for i in range(len(df)):
		update_indexes_temp(root_node,df.iloc[i],i)


def pruneTree(node, df): #prunes tree in for 1 go for all the parent of leaf nodes
	non_leaf_children = []
	for child in node.children:
		if child.finalVal == None:
			non_leaf_children.append(child)
	if non_leaf_children!=[]:
		for X in non_leaf_children:
			pruneTree(X, df)
	else:
		useful_df = df.iloc[node.indexes_array]
		if len(useful_df)==0:
			return
		num_pos = useful_df['Y'].astype(bool).sum(axis=0)
		num_neg = len(useful_df) - num_pos
		majority_pred_acc = (max(num_pos,num_neg)*1.0)/(num_pos+num_neg)
		pred_acc = predict_df(node, useful_df)[0]
		if majority_pred_acc >= pred_acc:
			node.finalVal = (num_neg,num_pos).index(max(num_pos,num_neg))
			node.children = None
			return
		else:
			return

def numNodes(node):
	if node.finalVal != None:
		return 1
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


#--------------------PART A----------------------------
root = treeNode(dataframe)
print("growing...")
start_time = time.time()
growTree(root,'global')
print("grow time = %f"%(time.time()-start_time))
#-----------------END A--------------------------

#************************PART B*************************

root2 = copy.deepcopy(root)
update_indexes(root2, df_val)
present_acc=0.0
past_acc=-1.0
num_nodes = []
val_acc = []
test_acc = []
train_acc = []
num_nodes.append(numNodes(root2))
val_acc.append(predict_df(root2, df_val)[0])
train_acc.append(predict_df(root2, df_train)[0])
test_acc.append(predict_df(root2, df_test)[0])
while(present_acc> past_acc):
	print("pruning")
	past_acc = present_acc
	pruneTree(root2, df_val)
	print("predicting")
	num_nodes.append(numNodes(root2))
	pred = predict_df(root2, df_val)
	val_acc.append(pred[0])
	train_acc.append(predict_df(root2, df_train)[0])
	test_acc.append(predict_df(root2, df_test)[0])
	present_acc = pred[0]

#************************END B****************************


#-------------PART C, also to do number of splits --
root = treeNode(dataframe)
print("growing...")
start_time = time.time()
growTree(root,'local')
print("grow time = %f"%(time.time()-start_time))
#-------------------END C----------------------

#*****************************************************PART D **********************************************************************

I,J,acc=0,0,0
for l in range(1,10):
	for m in range(2,10):
		tree = DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth = 10,min_samples_split=m,min_samples_leaf=l)
		tree.fit(x_train,y_train)
		pred = tree.score(x_val,y_val)
		if pred>acc:
			acc = pred
			I=m
			J=l
tree = DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth = 10,min_samples_split=I,min_samples_leaf=J)
tree.fit(x_train,y_train)
print("validation accuracy = %f"%tree.score(x_val,y_val))
print("train accuracy = %f"%tree.score(x_train,y_train))

#*************************************************************END D***************************************************************

#------------------------------PART E -----------------------------------------------
df2 = pd.DataFrame([\
	[0, 1, 0, 0, 0, -2, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 1, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 2, 2, 0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 3, 3, 0,  1,  1,  1,  1,  1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 4, 0, 0,  2,  2,  2,  2,  2,  2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 5, 0, 0,  3,  3,  3,  3,  3,  3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 6, 0, 0,  4,  4,  4,  4,  4,  4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 0, 0, 0,  5,  5,  5,  5,  5,  5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 0, 0, 0,  6,  6,  6,  6,  6,  6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 0, 0, 0,  7,  7,  7,  7,  7,  7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 0, 0, 0,  8,  8,  8,  8,  8,  8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	[0, 2, 0, 0, 0,  9,  9,  9,  9,  9,  9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
	], columns=df_test.columns)
df_test2 = df2.append(df_test)
df_train2 = df2.append(df_train)
df_val2 = df2.append(df_val)
dict = {-2:'a',-1:'b',0:'c',1:'d',2:'e',3:'f',4:'g',5:'h',6:'i',7:'j',8:'k',9:'l'}
sets = [df_test2,df_train2,df_val2]
for setwa in sets:
	for i in [[(3,3),(0,6)],[(4,4),(0,3)],[(6,11),(-2,9)]]:
		#[(startcol, endcol),(rangemin, rangemax)]
		startcol = i[0][0]
		endcol =   i[0][1]+1
		rangemin = i[1][0]
		rangemax = i[1][1]+1
		for col in range(startcol, endcol):
			for rang in range(rangemin,rangemax):
				column = 'X'+str(col)
				setwa[column][setwa[column]==rang] = dict[rang]
tree2 = DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth = 10,min_samples_split=9,min_samples_leaf=4)
df_test2 = pd.get_dummies(df_test2)
df_train2 = pd.get_dummies(df_train2)
df_val2 = pd.get_dummies(df_val2)
df_test2 = df_test2[12:]
df_train2 = df_train2[12:]
df_val2 = df_val2[12:]
x_train,y_train = df_train2.drop(columns=['Y']), df_train2['Y']
x_test, y_test = df_test2.drop(columns=['Y']), df_test2['Y']
x_val, y_val = df_val2.drop(columns=['Y']), df_val2['Y']
tree2.fit(x_train,y_train)
print("validation accuracy = %f"%tree2.score(x_val,y_val))
print("train accuracy = %f"%tree2.score(x_train,y_train))

#---------------------------------------------END E----------------------------------------------------------------------

#***********************PART F************************
tree3 = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0,bootstrap=False)
tree3.fit(x_train,y_train)
tree3.score(x_val,y_val)