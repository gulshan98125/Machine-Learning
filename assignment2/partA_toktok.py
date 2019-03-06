import json
import sys
import math
import pickle
import string
import re
import unicodedata
import numpy as np
from random import shuffle
import stemming_utilities as ul
from collections import Counter
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer().tokenize

fileName = ['data/train.json','data/test.json', 'data/train_full.json','data/test_full.json']
table = string.maketrans(string.punctuation,"                                ")
# table = string.maketrans(string.punctuation+'1234567890',"                                          ")

def json_reader(fname):
    for line in open(fname, mode="r"):
        yield json.loads(line)

def train(vocabulary, train_file_name, processed):
	print("training...")
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
	if processed==True:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1
			text_part_array = line['text'].split()
			words_per_label[int(label)-1] += len(text_part_array)

			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			if counter%5000==0:
				print('t_line no.: '+str(counter))
			counter+=1
	else:
		for line in json_reader(train_file_name):
			label = line['stars']
			per_class_count[int(label)-1] += 1

			# text_part = "".join(line['text'].split('\n'))
			# text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore').lower()
			# text_part = text_part.translate(table)
			# text_part_array = text_part.split()
			text_part_array = toktok(line['text'].lower())
			words_per_label[int(label)-1] += len(text_part_array)


			for word in text_part_array:
				count_dict[word][int(label)-1] += 1
			if counter%5000==0:
				print('t_line no.: '+str(counter))
			counter+=1

	#now we have count_dict and words_per_label. Lets update words probabilites per label
	for word in words_array:
		for i in range(5):
			count_dict[word][i] = (1.0*count_dict[word][i])/(1.0*words_per_label[i])

	#count_dict now contains probabilites of words per label (in logarithm)
	#----------------------------------------------
	return count_dict,words_per_label, per_class_count


def predict(words_string, count_dict, words_per_label, per_class_count,processed): #return class with max probability
	if processed:
		words_string_arr = words_string.split()
	else:
		# text_part = "".join(words_string.split('\n'))
		# text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore').lower()
		# text_part = text_part.translate(table)
		# words_string_arr = text_part.split()
		words_string_arr = toktok(words_string.lower())

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

def prediction_accuracy(file_name,a,b,c,processed, percentage_of_file):
	numLines = sum(1 for line in open(file_name))

	totalCount = 0
	correctCount=0
	for line in json_reader(file_name):
		prediction = predict(line['text'],a,b,c,processed)
		if prediction==line['stars']:
			correctCount+=1
		if totalCount%5000==0:
			print('p_line no.: '+str(totalCount))
		if totalCount > int(percentage_of_file*numLines):
			return (correctCount*1.0)/totalCount
			break
		totalCount+=1

	return (correctCount*1.0)/totalCount


#-----------for writing modal and loading it--------------------
# outfile = open('abc_train_full_stemmed.pickle','wb')
# pickle.dump((a,b,c),outfile)
# outfile.close()

# print("loading pickle")
# infile = open('models/abc_train_full_stemmed.pickle','rb')
# (a,b,c) = pickle.load(infile)
# infile.close()
#--------------------------------------------------------------



infile = open('vocabulary_train_processed.pickle','rb')
vocabulary = pickle.load(infile)
infile.close()
a,b,c = train(vocabulary, 'data/train.json', False)

print("predicting")
print prediction_accuracy('data/test.json',a,b,c, False,1)



# if __name__ == "__main__":
# 	train_data_path = str(sys.argv[1])
# 	test_data_path = str(sys.argv[2])
# 	part_num = str(sys.argv[3])
# 	pre_processing = str(sys.argv[3])

# 	if part_num=='a':
# 		if pre_processing is not None:
# 			infile = open('vocabulary/train_processed.pickle','rb')
# 			vocabulary = pickle.load(infile)
# 			infile.close()
# 			