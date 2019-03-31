import json
import sys
import math
import cPickle as pickle
import string
import unicodedata
from nltk import bigrams, pos_tag
from nltk.tokenize import ToktokTokenizer
import stemming_utilities as ul
# from nltk.tokenize import word_tokenize
# toktok = ToktokTokenizer().tokenize
from nltk.tag.perceptron import PerceptronTagger
tagger=PerceptronTagger()

# fileName = ['data/train.json','data/test.json', 'data/train_full.json','data/test_full.json']
table = string.maketrans(string.punctuation,'                                ')

def json_reader(fname):
    for line in open(fname, mode="r"):
        yield json.loads(line)

def create_vocabulary(train_file_name, stemmed, bigram):
	print("creating vocabulary")
	words_array = set()
	new_dict = {}
	counter = 0
	
	if stemmed==True:
		if bigram==True:
			with open('bigram_stemmed_'+str(train_file_name.split('/')[-1].split('.')[0])+'.json', mode="w") as fp:
				for line in json_reader(train_file_name):
					dictionary = {}
					text_part = "".join(line['text'].split('\n'))
					text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore')
					text_part = text_part.translate(table)
					text_part_array = ul.getStemmedDocuments(text_part)
					bigram_array = list(bigrams(text_part_array))
					dictionary['text'] = " ".join(text_part_array)
					dictionary['stars'] = line['stars']
					json.dump(dictionary, fp)
					fp.write("\n")
					for word in bigram_array:
						words_array.add(word)
					if counter%3000==0:
						print(counter)
					counter+=1
		
			outfile = open('vocabulary_'+str(train_file_name.split('/')[-1].split('.')[0])+'_bigram_stemmed.pickle','wb')
			pickle.dump(words_array,outfile,-1)
			outfile.close()
		else:
			with open('stemmed_'+str(train_file_name.split('/')[-1].split('.')[0])+'.json', mode="w") as fp:
				for line in json_reader(train_file_name):
					# dictionary = {}
					text_part = "".join(line['text'].split('\n'))
					text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore')
					text_part = text_part.translate(table)
					text_part_array = ul.getStemmedDocuments(text_part)
					# dictionary['text'] = " ".join(text_part_array)
					# dictionary['stars'] = line['stars']
					# json.dump(dictionary, fp)
					# fp.write("\n")
					for word in text_part_array:
						words_array.add(word)
					if counter%3000==0:
						print(counter)
					counter+=1
		
			outfile = open('vocabulary_'+str(train_file_name.split('/')[-1].split('.')[0])+'_stemmed.pickle','wb')
			pickle.dump(words_array,outfile,-1)
			outfile.close()

	else:
		if bigram==True:
			with open('bigram_processed_'+str(train_file_name.split('/')[-1].split('.')[0])+'.json', mode="w") as fp:
				for line in json_reader(train_file_name):
					# dictionary = {}
					text_part = "".join(line['text'].split('\n'))
					text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore')
					text_part = text_part.translate(table)
					bigram_array = list(bigrams(text_part.split()))
					# dictionary['text'] = text_part
					# dictionary['stars'] = line['stars']
					# json.dump(dictionary, fp)
					# fp.write("\n")
					for word in bigram_array:
						try:
							new_dict[word] += 1
						except:
							new_dict[word] = 1
						# words_array.add(word)
					if counter%5000==0:
						print(counter)
					counter+=1
			for x in new_dict:
				if new_dict[x] >1:
					words_array.add(x)
			outfile = open('vocabulary_'+str(train_file_name.split('/')[-1].split('.')[0])+'_bigram_processed.pickle','wb')
			pickle.dump(words_array,outfile,-1)
			outfile.close()
		else:
			with open('processed_'+str(train_file_name.split('/')[-1].split('.')[0])+'.json', mode="w") as fp:
				for line in json_reader(train_file_name):
					# dictionary = {}
					text_part = "".join(line['text'].split('\n'))
					text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore').lower()
					text_part = text_part.translate(table)
					# dictionary['text'] = text_part
					# dictionary['stars'] = line['stars']
					# json.dump(dictionary, fp)
					# fp.write("\n")
					text_part_array = text_part.split()
					# text_part_array = tagger.tag(text_part_array)
					# text_part_array = toktok(line['text'].lower())
					# text_part_array = re.sub("[^\w]", " ", line['text'].lower()).split()
					for word in text_part_array:
						words_array.add(word)
					print(counter)
					counter+=1

			outfile = open('vocabulary_'+str(train_file_name.split('/')[-1].split('.')[0])+'_processed.pickle','wb')
			pickle.dump(words_array,outfile,-1)
			outfile.close()
	print "done creating vocabulary"

create_vocabulary('data/parta/data/demo_train.json', True, False)