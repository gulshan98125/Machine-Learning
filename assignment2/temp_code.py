def train_test_with_split(vocabulary, train_file_name, processed, train_split_percent):
	numLines = sum(1 for line in open(train_file_name))
	split_arr = np.arange(numLines)
	shuffle(split_arr)
	train_part_arr = split_arr[:int(train_split_percent*numLines)]
	test_part_arr = split_arr[int(train_split_percent*numLines):]
	
	counter = 0
	words_array = vocabulary
	per_class_count = [0.0, 0.0, 0.0, 0.0, 0.0]
	count_dict = {}
	for word in words_array:
		count_dict[word] = [1,1,1,1,1]
	#**********************************************
	words_per_label = [len(words_array)]*5

	#training data---------------------------------
	new_test_file = 'test_from_'+str(train_file_name.split('/')[-1].split('.')[0])+'.json'
	with open(new_test_file, mode="w") as fp:
		if processed==True:
			for line in json_reader(train_file_name):
				if counter not in test_part_arr:		#train using that line
					label = line['stars']
					per_class_count[int(label)-1] += 1
					text_part_array = line['text'].split()
					words_per_label[int(label)-1] += len(text_part_array)

					for word in text_part_array:
						count_dict[word][int(label)-1] += 1
				else:										# dump that line for testing
					json.dump(line, fp)
					fp.write("\n")
				counter+=1
		else:
			for line in json_reader(train_file_name):
				if counter not in test_part_arr:
					label = line['stars']
					per_class_count[int(label)-1] += 1

					text_part = "".join(line['text'].split('\n'))
					text_part = unicodedata.normalize('NFKD', text_part).encode('ascii','ignore').lower()
					text_part = text_part.translate(table)
					text_part_array = text_part.split()
					words_per_label[int(label)-1] += len(text_part_array)

					for word in text_part_array:
						count_dict[word][int(label)-1] += 1
				else:
					json.dump(line, fp)
					fp.write("\n")
				counter+=1
	print("predicting accuracy wait...")
	print ("accuracy = "+ str(prediction_accuracy(new_test_file,count_dict,words_per_label, per_class_count, processed)*100)+"%")
