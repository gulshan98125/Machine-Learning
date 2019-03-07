import json
import sys
import math
import cPickle as pickle
import random
import string
from nltk import bigrams
import unicodedata
import numpy as np
from random import shuffle
import stemming_utilities as ul
from collections import Counter

fileName = ['data/train.json','data/test.json', 'data/train_full.json','data/test_full.json']
table = string.maketrans(string.punctuation," "*32)
# table = string.maketrans(string.punctuation+'1234567890',"                                          ")

def json_reader(fname):
    for line in open(fname, mode="r"):
        yield json.loads(line)

def bigram_train(vocabulary, train_file_name, processed):
	print("training, "),
	numLines = sum(1 for line in open(train_file_name))
	#generating count dict for every word*********
	words_array = vocabulary
	per_class_count = [0.0, 0.0, 0.0, 0.0, 0.0]
	count_dict = {}
	for word in words_array:
		count_dict[word] = [1,1,1,1,1]
	#**********************************************
	words_per_label = [len(words_array)]*5
	#training data---------------------------------
	counter = 0
	print("total lines = "+ str(numLines/1000)+'k')
	print('training_line no.: '),
	if processed==True:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1
			text_part_array = list(bigrams(line['text'].split()))
			words_per_label[int(label)-1] += len(text_part_array)

			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			counter+=1
			if counter%10000==0:
				print(str(counter/1000)+'k'),
	else:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1

			text_part = "".join(line['text'].split('\n'))
			text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore').lower()
			text_part = text_part.translate(table)
			text_part_array = list(bigrams(text_part.split()))
			words_per_label[int(label)-1] += len(text_part_array)


			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			counter+=1
			if counter%10000==0:
				print(str(counter/1000)+'k'),

	#now we have count_dict and words_per_label. Lets update words probabilites per label
	for word in words_array:
		for i in range(5):
			count_dict[word][i] = (1.0*count_dict[word][i])/(1.0*words_per_label[i])

	#count_dict now contains probabilites of words per label (in logarithm)
	#----------------------------------------------
	print("\n")
	return count_dict,words_per_label, per_class_count

def stemmed_train(vocabulary, train_file_name, processed):
	print("training, "),
	numLines = sum(1 for line in open(train_file_name))
	#generating count dict for every word*********
	words_array = vocabulary
	per_class_count = [0.0, 0.0, 0.0, 0.0, 0.0]
	count_dict = {}
	for word in words_array:
		count_dict[word] = [1,1,1,1,1]
	#**********************************************
	words_per_label = [len(words_array)]*5
	#training data---------------------------------
	counter = 0
	print("total lines = "+ str(numLines/1000)+'k')
	print('training_line no.: '),
	if processed==True:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1
			text_part_array = line['text'].split()
			words_per_label[int(label)-1] += len(text_part_array)

			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			counter+=1
			if counter%10000==0:
				print(str(counter/1000)+'k'),
	else:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1

			text_part = "".join(line['text'].split('\n'))
			text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore')
			text_part = text_part.translate(table)
			text_part_array = ul.getStemmedDocuments(text_part)
			words_per_label[int(label)-1] += len(text_part_array)


			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			counter+=1
			if counter%1000==0:
				print(str(counter/1000)+'k'),

	#now we have count_dict and words_per_label. Lets update words probabilites per label
	for word in words_array:
		for i in range(5):
			count_dict[word][i] = (1.0*count_dict[word][i])/(1.0*words_per_label[i])

	#count_dict now contains probabilites of words per label (in logarithm)
	#----------------------------------------------
	print("\n")
	return count_dict,words_per_label, per_class_count

def train(vocabulary, train_file_name, processed):
	print("training, "),
	numLines = sum(1 for line in open(train_file_name))
	#generating count dict for every word*********
	words_array = vocabulary
	per_class_count = [0.0, 0.0, 0.0, 0.0, 0.0]
	count_dict = {}
	for word in words_array:
		count_dict[word] = [1,1,1,1,1]
	#**********************************************
	words_per_label = [len(words_array)]*5
	#training data---------------------------------
	counter = 0
	print("total lines = "+ str(numLines/1000)+'k')
	print('training_line no.: '),
	if processed==True:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1
			text_part_array = line['text'].split()
			words_per_label[int(label)-1] += len(text_part_array)

			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			counter+=1
			if counter%10000==0:
				print(str(counter/1000)+'k'),
	else:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1

			text_part = "".join(line['text'].split('\n'))
			text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore').lower()
			text_part = text_part.translate(table)
			text_part_array = text_part.split()
			words_per_label[int(label)-1] += len(text_part_array)


			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			counter+=1
			if counter%10000==0:
				print(str(counter/1000)+'k'),

	#now we have count_dict and words_per_label. Lets update words probabilites per label
	for word in words_array:
		for i in range(5):
			count_dict[word][i] = (1.0*count_dict[word][i])/(1.0*words_per_label[i])

	#count_dict now contains probabilites of words per label (in logarithm)
	#----------------------------------------------
	print("\n")
	return count_dict,words_per_label, per_class_count


def predict(words_string, count_dict, words_per_label, per_class_count,processed,method): #method = normal, stemmed, bigram
	if processed:
		if method=='bigram':
			words_string_arr = list(bigrams(words_string.split()))
		else:
			words_string_arr = words_string.split()
	else:
		text_part = "".join(words_string.split('\n'))
		text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore').lower()
		text_part = text_part.translate(table)

		if method=='stemmed':
			words_string_arr = ul.getStemmedDocuments(text_part)
		elif method=='normal':
			words_string_arr = text_part.split()
		elif method=='bigram':
			words_string_arr = list(bigrams(text_part.split()))

	per_class_prob = [0.0, 0.0, 0.0, 0.0, 0.0]
	for word in words_string_arr:
		for i in range(5):
			try:
				per_class_prob[i] += math.log(count_dict[word][i])
			except:
				# per_class_prob[i] -= math.log((per_class_count[i]+5)*1.0)
				per_class_prob[i] -= math.log(words_per_label[i]*1.0)

	total_class_count = sum(per_class_count)
	# total_class_count = 1.0*sum(per_class_count) - len(per_class_count)*5.0

	for i in range(5):
		# per_class_prob[i] += math.log(1.0*(per_class_count[i]-5))-math.log(1.0*total_class_count)
		per_class_prob[i] += math.log(1.0*per_class_count[i])-math.log(1.0*total_class_count)

	return (per_class_prob.index(max(per_class_prob))+1)

def prediction_accuracy(file_name,a,b,c,processed, percentage_of_file,method): #method = normal, stemmed, bigram
	numLines = sum(1 for line in open(file_name))

	totalCount = 0
	correctCount=0
	print("total lines = "+ str(int(percentage_of_file*numLines)/1000)+'k')
	print('prediction line no.: '),
	for line in json_reader(file_name):
		prediction = predict(line['text'],a,b,c,processed,method)
		if prediction==line['stars']:
			correctCount+=1
		if totalCount%10000==0:
			print(str(totalCount/1000)+'k'),
		if totalCount > int(percentage_of_file*numLines):
			print("\n")
			return (correctCount*1.0)/totalCount
			break
		totalCount+=1
	print("\n")
	return (correctCount*1.0)/totalCount

def random_prediction_accuracy(file_name,percentage_of_file):
	numLines = sum(1 for line in open(file_name))
	totalCount = 0
	correctCount=0
	for line in json_reader(file_name):
		if line['stars']==random.randint(1,5):
			correctCount+=1
		if totalCount > int(percentage_of_file*numLines):
			return (correctCount*1.0)/totalCount
			break
		totalCount+=1
	return (correctCount*1.0)/totalCount

def majority_prediction_accuracy(file_name,percentage_of_file):
	totalCount = 0
	correctCount=0
	numLines = sum(1 for line in open(file_name))
	class_count = [0.0, 0.0, 0.0, 0.0, 0.0]	
	for line in json_reader(file_name):
		class_count[int(line['stars'])-1] += 1

	majority_class = class_count.index(max(class_count))+1
	for line in json_reader(file_name):
		if line['stars']==majority_class:
			correctCount+=1
		if totalCount > int(percentage_of_file*numLines):
			return (correctCount*1.0)/totalCount
			break
		totalCount+=1
	return (correctCount*1.0)/totalCount

def get_confusion_matrix(file_name,a,b,c,processed):
	matrix = np.zeros((5,5))
	matrix = matrix.tolist()
	numLines = sum(1 for line in open(file_name))
	totalCount = 0
	print("total lines = "+ str(numLines/1000)+'k')
	print('prediction line no.: '),
	for line in json_reader(file_name):
		prediction = predict(line['text'],a,b,c,processed,'normal')
		matrix[int(prediction)-1][int(line['stars'])-1]+=1
		if totalCount%10000==0:
			print(str(totalCount/1000)+'k'),
		totalCount+=1
	print("\n")
	return np.array(matrix)
#-----------for writing modal and loading it--------------------
# outfile = open('abc_train_full_stemmed.pickle','wb')
# pickle.dump((a,b,c),outfile)
# outfile.close()

# print("loading pickle")
# infile = open('models/abc_train_full_stemmed.pickle','rb')
# (a,b,c) = pickle.load(infile)
# infile.close()
#--------------------------------------------------------------



# infile = open('vocabulary_train_processed.pickle','rb')
# vocabulary = pickle.load(infile)
# infile.close()
# a,b,c = train(vocabulary, 'data/train.json', False)

# print("predicting")
# print prediction_accuracy('data/test.json',a,b,c, False,1)



if __name__ == "__main__":
	train_data_path = str(sys.argv[1])
	test_data_path = str(sys.argv[2])
	part_num = str(sys.argv[3])
	if 5==len(sys.argv):
		pre_processing = bool(str(sys.argv[4]))
	else:
		pre_processing= False

	if part_num=='a':
		infile = open('vocabulary/train.pickle','rb')
		vocabulary = pickle.load(infile)
		infile.close()

		a,b,c = train(vocabulary, train_data_path, pre_processing)
		print("predicting on test, "),
		test_accuracy = prediction_accuracy(test_data_path,a,b,c,pre_processing,1,'normal')*100
		print("predicting on train, "),
		train_accuracy = prediction_accuracy(train_data_path,a,b,c,pre_processing,0.2,'normal')*100
		print("test accuracy = "+ str(test_accuracy) +" %" )
		print("train accuracy = "+ str(train_accuracy) +" %" )

	elif part_num=='b':
		print("random predicting on test")
		random_accuracy1 = random_prediction_accuracy(test_data_path,1)
		print("random predicting on train\n")
		random_accuracy2 = random_prediction_accuracy(train_data_path,0.2)
		print("random test accuracy = "+ str(random_accuracy1*100) +" %" )
		print("random train accuracy = "+ str(random_accuracy2*100) +" %\n" )

		print("majority predicting on test")
		majority_accuracy1 = majority_prediction_accuracy(test_data_path,1)

		print("majority predicting on train\n")
		majority_accuracy2 = majority_prediction_accuracy(train_data_path,0.2)
		print("majority test accuracy = "+ str(majority_accuracy1*100) +" %" )
		print("majority train accuracy = "+ str(majority_accuracy2*100) +" %" )

	elif part_num=='c':
		infile = open('vocabulary/train.pickle','rb')
		vocabulary = pickle.load(infile)
		infile.close()
		a,b,c = train(vocabulary, train_data_path, pre_processing)
		print("obtaining confusion matrix...")
		confusion_matrix = get_confusion_matrix(test_data_path,a,b,c, pre_processing)
		print confusion_matrix

	elif part_num=='d':
		infile = open('vocabulary/train_stemmed.pickle','rb')
		vocabulary = pickle.load(infile)
		infile.close()

		a,b,c = stemmed_train(vocabulary, train_data_path, pre_processing)
		print("predicting on test, "),
		test_accuracy = prediction_accuracy(test_data_path,a,b,c,pre_processing,1,'stemmed')*100
		print("predicting on train, "),
		train_accuracy = prediction_accuracy(train_data_path,a,b,c,pre_processing,0.2,'stemmed')*100
		print("stemmed test accuracy = "+ str(test_accuracy) +" %" )
		print("stemmed train accuracy = "+ str(train_accuracy) +" %" )

	elif part_num=='e':
		infile = open('vocabulary/train_bigram.pickle','rb')
		vocabulary = pickle.load(infile)
		infile.close()

		a,b,c = bigram_train(vocabulary, train_data_path, pre_processing)
		print("predicting on test, "),
		test_accuracy = prediction_accuracy(test_data_path,a,b,c,pre_processing,1,'bigram')*100
		print("predicting on train, "),
		train_accuracy = prediction_accuracy(train_data_path,a,b,c,pre_processing,0.2,'bigram')*100
		print("bigrams test accuracy = "+ str(test_accuracy) +" %" )
		print("bigrams train accuracy = "+ str(train_accuracy) +" %" )




