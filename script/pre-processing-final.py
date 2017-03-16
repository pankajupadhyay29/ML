#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import csv
import re

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
#### Functions ######

def tf():
	#pickle.dump(vect.vocabulary_, open(dictionary_filepath, 'w'))

	# Load the vocabulary
	#vocabulary_to_load = pickle.load(open(dictionary_filepath, 'r'))
	
	return TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english', decode_error = 'ignore')#, vocabulary=vocabulary_to_load)

def removeHtmlTags(str):
	return re.compile(r'<[^>]+>').sub('', str)
	
def get_tfidf_score(dataDictionary):
	for id, text in dataDictionary.iteritems():
		dataDictionary[id] = "".join(text)						
		
	corpus = []
	for id, dataDictionary in sorted(dataDictionary.iteritems(), key=lambda t: int(t[0])):
		corpus.append(dataDictionary)
		
	### calculate ###			
	tfidf_matrix =  tf().fit_transform(corpus)
	feature_names = tf().fit(corpus).get_feature_names() 
	
	dense = tfidf_matrix.todense()
	title = dense[0].tolist()[0]
	phrase_scores = [pair for pair in zip(range(0, len(title)), title) if pair[1] > 0]
 
			
	sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
	
	return [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores]
	
def write_tdidf_to_file(data, fileName, append = True):
	filePath = os.path.join(os.getcwd(), fileName)
	if append is not True and os.path.isfile(filePath):
		os.remove(filePath)
	
	with open(filePath, 'wb') as fWriter:
		writer = csv.writer(fWriter)#UnicodeWriter(fWriter)
		writer.writerow(["Phrase", "Score"])
		
		#writer.writerows(data)
		for phrase, score in data:						
			writer.writerow([unicode(phrase).encode('utf8'), score])

def startProcessing ():
	titles = defaultdict(list)
	titlesTFIDF = []
	descriptions = defaultdict(list)
	descriptionsTFIDF = []
	tags = defaultdict(list)
	tagsTFIDF = []
	
	rawDataDirectory = os.getcwd() + "\\Raw Data"
	cleanDataDirectory = os.getcwd() + "\\Clean Data"

	for root,dirs,files in os.walk(rawDataDirectory):
		for file in files:
			if ".csv" not in file:
				continue
				
			with open(os.path.join(root, file), 'r') as fReader:							
				reader = csv.reader(fReader)
				reader.next()
				for row in reader:
					titles[row[0]].append(row[1])
					descriptions[row[0]].append(row[2])
					tags[row[0]].append(row[3])
			
			titlesTFIDF = get_tfidf_score(titles)
			descriptionsTFIDF = get_tfidf_score(descriptions)
			tagsTFIDF = get_tfidf_score(tags)
			
			fileName = os.path.join(root.replace(rawDataDirectory, cleanDataDirectory), file)
			write_tdidf_to_file(titlesTFIDF,fileName.replace(".csv", "_titlesTFIDF.csv"), append = False)			
			write_tdidf_to_file(descriptionsTFIDF, fileName.replace(".csv", "_descriptionsTFIDF.csv"), append = False)			
			write_tdidf_to_file(tagsTFIDF, fileName.replace(".csv", "_tagsTFIDF.csv"), append = False)			
			

		

	
#### Function End #####
	
	
####  Processing Start ####

startProcessing()