# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import nltk
import re
import string
import math
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from subprocess import check_output
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import operator

swords1 = stopwords.words('english')
punctuations = string.punctuation
wordnet_lemmatizer = WordNetLemmatizer()

print("Start")

def data_clean(data):
	#print('Cleaning data')
	data = data.apply(lambda x: x.lower())
	data = data.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
	data = data.apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))
	data = data.apply(lambda i: ''.join(i.strip(punctuations))  )
	#print('tokenize')
	data = data.apply(lambda x: word_tokenize(x))

	#Select only the nouns
	is_noun = lambda pos: pos[:2] == 'NN' #or pos[:2] == 'ADJ' 
	for i in range(len(data)):
		data[i] = [word for (word, pos) in nltk.pos_tag(data[i]) if is_noun(pos)]
	
	#print('Remove stopwords')
	data = data.apply(lambda x: [i for i in x if i not in swords1 if len(i)>2])
	#print('minor clean some words')
	data = data.apply(lambda x: [i.split('/') for i in x] )
	data = data.apply(lambda x: [i for y in x for i in y])
	#print('Lemmatizing')
	
	data = data.apply(lambda x: [wordnet_lemmatizer.lemmatize(i) for i in x])
	data = data.apply(lambda x: [i for i in x if len(i)>2])
	return(data)

	
def get_combined_frequency(content, title):		
	frequency = []
	inverse_frequency = {}
	length = 0
	if content is not None:
		length = len(content)
	else:
		length = len(title)
	for i in range(length):
		word_count = {}
		important = {}
		if title is not None:
			for word in title[i]:
				if word in word_count:
					word_count[word] = word_count[word] + 100
				else:
					word_count[word] = 10
					important[word] = True
		if content is not None:
			for word in content[i]:
				if word in word_count:
					if word in important:
						word_count[word] = word_count[word] + 50
					else:
						word_count[word] = word_count[word] + 5
				else:
					word_count[word] = 1
				
		for word in word_count:
			if word in inverse_frequency:
				inverse_frequency[word] = inverse_frequency[word] + 1
			else:
				inverse_frequency[word] = 1		
		frequency.append(word_count)
	return (frequency, inverse_frequency)

def get_tfidf(frequency, inverse_frequency):
	tfidf_distribution = []
	#word_tfidf = []
	for document in frequency:
		if document == {}:
			continue
		max_frequency = sorted(document.items(), key=operator.itemgetter(1), reverse=True)[0][1]
		for word in document:
			#document[word] = document[word]/(max_frequency + 0.0)*np.log(len(frequency)/(inverse_frequency[word]+0.))
			document[word] = document[word]/(max_frequency + 0.0)*np.log(len(frequency)/(1.0 + inverse_frequency[word]))
			#tfidf_distribution.append(document[word])
			tfidf_distribution.append((word, document[word]))
	return tfidf_distribution

def plot_distribution(data, fileName, format):
	val = plt.plot(np.log(list(filter(lambda x: x is not 0, data))))
	plt.savefig(fileName, format=format)
	plt.close()
	
	return val
	
def plot_bar(data, fileName, format):
	label = [x[0] for x in data]
	score = [x[1] for x in data]
	x_pos = np.arange(len(label)) 

	# calculate slope and intercept for the linear trend line
	slope, intercept = np.polyfit(x_pos, score, 1)
	trendline = intercept + (slope * x_pos)

	#plt.plot(x_pos, trendline, color='red', linestyle='--')    
	plt.bar(x_pos, score, align='center')
	plt.xticks(x_pos, label, rotation=90)	
	plt.ylabel('Score')
	plt.tight_layout()
	plt.savefig(fileName, format=format)
	plt.close()
	
def process_data(data):
	ids = data.id
	content = data_clean(data.content)
	title = data_clean(data.title)
	#tags = data_clean(data.tags)

	#frequency_title, inverse_frequency_title = get_combined_frequency(None, title)
	#frequency_content, inverse_frequency_content = get_combined_frequency(content, None)
	frequency_title_content, inverse_frequency_title_content = get_combined_frequency(content, title)
	#frequency_tags, inverse_frequency_tags = get_combined_frequency(None, tags)
	
	#tfidf_title = get_tfidf(frequency_title, inverse_frequency_title)
	#tfidf_content = get_tfidf(frequency_content, inverse_frequency_content)
	tfidf_title_content = get_tfidf(frequency_title_content, inverse_frequency_title_content)
	#tfidf_tags = get_tfidf(frequency_tags, inverse_frequency_tags)	
			
	#plot_bar(tfidf_title[0:50], "random_title.svg", format="svg")
	#plot_bar(tfidf_content[0:50], "random_content.svg", format="svg")
	#plot_bar(tfidf_tags[0:50], "random_tags.svg", format="svg")
	#plot_bar(tfidf_title_content[0:50], "random_title_content.svg", format="svg")
	#plot_bar(sorted(tfidf_title, key=lambda x: x[1])[0:50], "bottom_title.svg", format="svg")
	#plot_bar(sorted(tfidf_content, key=lambda x: x[1])[0:50], "bottom_content.svg", format="svg")
	#plot_bar(sorted(tfidf_tags, key=lambda x: x[1])[0:50], "bottom_tags.svg", format="svg")
	#plot_bar(sorted(tfidf_title_content, key=lambda x: x[1])[0:50], "bottom_title_content.svg", format="svg")
	#plot_bar(sorted(tfidf_title, key=lambda x: x[1], reverse=True)[0:50], "top_title.svg", format="svg")
	#plot_bar(sorted(tfidf_content, key=lambda x: x[1], reverse=True)[0:50], "top_content.svg", format="svg")
	#plot_bar(sorted(tfidf_tags, key=lambda x: x[1], reverse=True)[0:50], "top_tags.svg", format="svg")
	#plot_bar(sorted(tfidf_title_content, key=lambda x: x[1], reverse=True)[0:50], "top_title_content.svg", format="svg")
	
	return (frequency_title_content, tfidf_title_content)

def getFScore(prediction,tags):
    if len(prediction) == 0 or len(tags) == 0:
        return 0.0    
    corrects = 0
    for p in prediction:
        if p in tags:
            corrects = corrects + 1
    
    precision = corrects / (len(prediction) + 0.)
    recall = corrects / (len(tags) + 0.)
    if precision == 0 or recall == 0:
        return 0.0     
    return 2*precision*recall/(precision + recall)
	
def predict_tag(data, fileName):
	predicted_tag, distribution = process_data(data)
	distribution_lines = plot_distribution(sorted([tfidf[1] for tfidf in distribution]), fileName+"_tfidf_distribution.svg", format="svg")
	#print("plotted")
	output = []
	max_tfidf = math.ceil(distribution_lines[0].axes.get_ylim()[1])
	print (max_tfidf)
	accuracy_by_id = []
	scores = []
	for i in range(0,len(data)):
		prediction = sorted(predicted_tag[i], key=predicted_tag[i].get, reverse=True)[0: max_tfidf]		
		id = data.id[i]
		if 'tags' in data:
			tags = data.tags[i]#.replace('-', ' ')
			tagsClean  = set([wordnet_lemmatizer.lemmatize(word) for word in tags.split()])
			f1_score = getFScore(prediction, tagsClean)
			scores.append(f1_score)
			accuracy_by_id.append((id, f1_score))
			
			output.append([id, ' '.join(prediction), tags, f1_score])			
		else:									
			#print("Test")			
			output.append([id, data.title[i],data.content[i], ' '.join(prediction)])
			
	if 'tags' in data:
		print(fileName)
		print(np.average(scores))
		plot_distribution(scores, fileName+"_accuracy.svg", format="svg")
				
		pd.DataFrame(data=output,columns = ['id','predicted_tags', 'tags', 'f1_score']).to_csv(os.getcwd().replace('script','output')+"/"+fileName + '_output.csv', index=False)	
	else:
		pd.DataFrame(data=output,columns = ['id','title', 'content', 'tags']).to_csv(os.getcwd().replace('script','output')+"/"+fileName + '_output.csv', index=False)
			
	
def start_processing():
	#print(os.getcwd())
	rawDataDirectory = os.getcwd().replace('script','input'	)
	#print(rawDataDirectory)
	for root,dirs,files in os.walk(rawDataDirectory):		
		for file in files:
			if '.csv' not in file:
				continue

			data = pd.read_csv(os.path.realpath(file).replace('script','input'), encoding = 'latin-1')
			predict_tag(data,os.path.splitext(os.path.basename(file))[0])
	
start_processing()
print("End")