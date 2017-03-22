# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import nltk
import re
import string
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

print("Start")

def data_clean(data):
	print('Cleaning data')
	data = data.apply(lambda x: x.lower())
	data = data.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
	data = data.apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))
	data = data.apply(lambda i: ''.join(i.strip(punctuations))  )
	print('tokenize')
	data = data.apply(lambda x: word_tokenize(x))

	#Select only the nouns
	is_noun = lambda pos: pos[:2] == 'NN' 
	for i in range(len(data)):
		data[i] = [word for (word, pos) in nltk.pos_tag(data[i]) if is_noun(pos)]
	
	print('Remove stopwords')
	data = data.apply(lambda x: [i for i in x if i not in swords1 if len(i)>2])
	print('minor clean some words')
	data = data.apply(lambda x: [i.split('/') for i in x] )
	data = data.apply(lambda x: [i for y in x for i in y])
	print('Lemmatizing')
	wordnet_lemmatizer = WordNetLemmatizer()
	data = data.apply(lambda x: [wordnet_lemmatizer.lemmatize(i) for i in x])
	data = data.apply(lambda x: [i for i in x if len(i)>2])
	return(data)

	
def get_conbined_frequency(content, title):
		
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
						word_count[word] = word_count[word] + 1
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
			document[word] = document[word]/(max_frequency + 0.0)*np.log(len(frequency)/(inverse_frequency[word]+0.))
			#tfidf_distribution.append(document[word])
			tfidf_distribution.append((word, document[word]))
	return tfidf_distribution

def plot_bar(data, fileName, format):
	label = [x[0] for x in data]
	score = [x[1] for x in data]
	x_pos = np.arange(len(label)) 

	# calculate slope and intercept for the linear trend line
	slope, intercept = np.polyfit(x_pos, score, 1)
	trendline = intercept + (slope * x_pos)

	#plt.plot(x_pos, trendline, color='red', linestyle='--')    
	plt.bar(x_pos, score, align='center')
	plt.xticks(x_pos, label, rotation=80)	
	plt.ylabel('Score')
	plt.tight_layout()
	plt.savefig(fileName, format=format)
	plt.close()
	
def process_data(data):
	content = data_clean(data.content)
	title = data_clean(data.title)
	tags = data_clean(data.tags)

	frequency_title, inverse_frequency_title = get_conbined_frequency(None, title)
	frequency_content, inverse_frequency_content = get_conbined_frequency(content, None)
	frequency_title_content, inverse_frequency_title_content = get_conbined_frequency(content, title)
	frequency_tags, inverse_frequency_tags = get_conbined_frequency(None, tags)
	
	tfidf_title = get_tfidf(frequency_title, inverse_frequency_title)
	tfidf_content = get_tfidf(frequency_content,inverse_frequency_content)
	tfidf_title_content = get_tfidf(frequency_title_content,inverse_frequency_title_content)
	tfidf_tags = get_tfidf(frequency_tags, inverse_frequency_tags)
	#print(tfidf_title[0:20])
	plot_bar(tfidf_title[0:50], "random_title.svg", format="svg")
	plot_bar(tfidf_content[0:50], "random_content.svg", format="svg")
	plot_bar(tfidf_tags[0:50], "random_tags.svg", format="svg")
	plot_bar(tfidf_title_content[0:50], "random_title_content.svg", format="svg")
	plot_bar(sorted(tfidf_title, key=lambda x: x[1])[0:50], "bottom_title.svg", format="svg")
	plot_bar(sorted(tfidf_content, key=lambda x: x[1])[0:50], "bottom_content.svg", format="svg")
	plot_bar(sorted(tfidf_tags, key=lambda x: x[1])[0:50], "bottom_tags.svg", format="svg")
	plot_bar(sorted(tfidf_title_content, key=lambda x: x[1])[0:50], "bottom_title_content.svg", format="svg")
	plot_bar(sorted(tfidf_title, key=lambda x: x[1], reverse=True)[0:50], "top_title.svg", format="svg")
	plot_bar(sorted(tfidf_content, key=lambda x: x[1], reverse=True)[0:50], "top_content.svg", format="svg")
	plot_bar(sorted(tfidf_tags, key=lambda x: x[1], reverse=True)[0:50], "top_tags.svg", format="svg")
	plot_bar(sorted(tfidf_title_content, key=lambda x: x[1], reverse=True)[0:50], "top_title_content.svg", format="svg")
	
	
def start_processing():
	biology = pd.read_csv("../input/biology.csv", encoding = 'latin-1')
	process_data(biology)

start_processing()
print("End")