#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import csv
import re

from collections import defaultdict

#### Functions ######

from sklearn.feature_extraction.text import TfidfVectorizer

def tf():
	#pickle.dump(vect.vocabulary_, open(dictionary_filepath, 'w'))

	# Load the vocabulary
	#vocabulary_to_load = pickle.load(open(dictionary_filepath, 'r'))
	
	return TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')#, vocabulary=vocabulary_to_load)

def removeHtmlTags(str):
	return re.compile(r'<[^>]+>').sub('', str)	

def startProcessing ():
	questions = defaultdict(list)
	
	rawDataDirectory = os.getcwd() + "\\Test Data"


	for root,dirs,files in os.walk(rawDataDirectory):
		for file in files:
			if ".csv" not in file:
				continue
				
			with open(os.path.join(root, file), 'r') as fReader:							
				reader = csv.reader(fReader)
				reader.next()
				for row in reader:
					questions[row[0]].append(row[1]+ removeHtmlTags(row[2]))
					#print content
					#print "*********************"
			
			for id, text in questions.iteritems():
				questions[id] = "".join(text)
				#print questions[id]
				#print "*********************"
			
			corpus = []
			for id, question in sorted(questions.iteritems(), key=lambda t: int(t[0])):
				corpus.append(question)
				
			### calculate ###			
			tfidf_matrix =  tf().fit_transform(corpus)
			feature_names = tf().fit(corpus).get_feature_names() 
			
			dense = tfidf_matrix.todense()
			question = dense[0].tolist()[0]
			phrase_scores = [pair for pair in zip(range(0, len(question)), question) if pair[1] > 0]
 
			
			sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
			for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores]:
			   print(phrase, score)
			

		

	
#### Function End #####
	
	
####  Processing Start ####

startProcessing()