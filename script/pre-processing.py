#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import csv
import re
import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
#from textblob import TextBlob as tb

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words('english'))
#print stop_words
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])#add panctuations to removal
#print stop_words

TAG_REMOVAL_REGX = re.compile(r'<[^>]+>')

poter_Stemmer = PorterStemmer()
snowball_Stemmer = SnowballStemmer("english")

def tf(word, list):
    return list.count(word) / len(list)

def n_containing(word, all_content):
    return sum(1 for list in all_content if word in list)

def idf(word, all_content):
    return math.log(len(all_content) / (1 + n_containing(word, all_content)))

def tfidf(word, list, all_content):
    return tf(word, list) * idf(word, all_content)

def make_unicode(input):
    if type(input) != unicode:
        input =  input.decode(encoding="utf-8", errors="ignore")
        return input
    else:
        return input
		
def make_ascii(input):
    input =  input.decode(encoding="ascii", errors="ignore")
    return input

		
def removeHtmlTags(str):
	return TAG_REMOVAL_REGX.sub('', str)	
	
def tokenizeString(val):	
	return word_tokenize(make_unicode(val))#(val.decode(encoding="utf-8", errors="ignore"))#(val.replace('[μm]','[I1/4M]'))

	
	
def removeStopWord(list):	
	clearList = []
	if list is not None:
		for word in list:
			if word.lower() not in stop_words:
				clearList.append(word)	
	return clearList	
	
	
	
def make_row_unicode(row):
	if list is not None:
		for index in range(len(row)):
			row[index] = make_unicode(row[index]) #unicode(row[index],encoding="utf-8", errors="ignore")#
	
	return row
	
def stem_word(words):
	stemmed_words = []
	for word in words:
		stemmed_words.append(snowball_Stemmer.stem(word))
	
	return stemmed_words
	
def getProcessedRow(row):
	row[2] = removeHtmlTags(row[2])
	#row = make_row_unicode(row)
	#print row[2]
	title_word_list = removeStopWord(stem_word(tokenizeString(row[1])))
	desc_word_list = removeStopWord(stem_word(tokenizeString(row[2])))
	tags = stem_word(tokenizeString(row[3]))
	return (title_word_list, desc_word_list, tags, (title_word_list + desc_word_list))

def calculateTFIDF(processed_rows, content):
	scroed_rows = []
	for row in processed_rows:
		scores = {word: tfidf(word, row[3], content) for word in row[3]}
		scroed_rows.append((row, scores))
		print(scores)
		print("################################")
	return scroed_rows
	
print('starting')

rawDataDirectory = os.getcwd() + "\\Test Data"

cleanDataDirectory = os.getcwd() + "\\Clean Data"
#print(rawDataDirectory)

for root,dirs,files in os.walk(rawDataDirectory):
    for file in files:
		if ".csv" not in file:
			continue
			
		with open(os.path.join(root, file), 'r') as fReader:
			#fWriter = open(os.path.join(root.replace(rawDataDirectory, cleanDataDirectory), file), 'w')
			try:
				all_processed_rows = []
				processed_content_doc = []
				reader = csv.reader(fReader)
				#writer = csv.writer(fWriter)
				for i, line in enumerate(reader):
					#print 'line[{}] = {}'.format(i, line)
					#writer.writerow(getProcessedRow(line))		
					#print i, file
					processed_data = getProcessedRow(line)
					print processed_data[2]
					print processed_data[3]					
					print "*************************************"
					all_processed_rows.append(processed_data)
					processed_content_doc.append(processed_data[3])
				(calculateTFIDF(all_processed_rows, processed_content_doc))
			finally:
				print 'done'
				#fWriter.close()
		
