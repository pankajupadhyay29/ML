# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import seaborn as sns
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
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet

swords1 = stopwords.words('english')
punctuations = string.punctuation
wordnet.MORPHOLOGICAL_SUBSTITUTIONS[NOUN].append(('ing', '')) # customization to remove ing from nouns
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
  is_noun = lambda pos: pos[:2] in ['NN', 'NNS', 'NNP', 'NNPS'] #or pos[:2] == 'ADJ' 
  for i in range(len(data)):
    data[i] = [word for (word, pos) in nltk.pos_tag(data[i]) if is_noun(pos)]
  
  #print('Remove stopwords')
  data = data.apply(lambda x: [i for i in x if i not in swords1 if len(i)>2])
  #print('minor clean some words')
  data = data.apply(lambda x: [i.split('/') for i in x] )
  data = data.apply(lambda x: [i for y in x for i in y])
  #print('Lemmatizing')

  print(wordnet.MORPHOLOGICAL_SUBSTITUTIONS)
  
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
    total_docs = len(frequency)
    #mean_frequency = np.mean()
    for word in document:
      frequency_value = document[word]
      inverse_frequency_Value =inverse_frequency[word]
      tf = frequency_value/max_frequency #frequency_value/len(document) #0.5+ (0.5*frequency_value/(max_frequency + 0.0))
      idf = np.log(total_docs/(1.0+inverse_frequency_Value)) #np.log((total_docs-inverse_frequency_Value)/inverse_frequency_Value) #inverse_frequency[word]#np.log(total_docs/(1.0 + inverse_frequency_Value))
      #document[word] = document[word]/(max_frequency + 0.0)*np.log(total_docs/(inverse_frequency_Value+0.))
      document[word] = tf * idf
      tfidf_distribution.append(document[word])
      #tfidf_distribution.append((word, document[word]))
  return tfidf_distribution

def plot_distribution(data, fileName, format):
  val = plt.plot(data)
  plt.savefig(fileName, format=format)
  plt.close()

  sns.distplot(data);
  sns.plt.savefig('sns_'+fileName, format=format)
  sns.plt.close()
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

  frequency_title_content, inverse_frequency_title_content = get_combined_frequency(content, title)
  
  tfidf_title_content = get_tfidf(frequency_title_content, inverse_frequency_title_content)
  
  return (frequency_title_content, tfidf_title_content)

def getFScore(prediction,tags):
    f1Score = 0
    precision = 0
    recall = 0
    if len(prediction) == 0 or len(tags) == 0:
        f1Score = 0.0
    else:
      corrects = 0
      for p in prediction:
          if p in tags:
              corrects = corrects + 1
      
      precision = corrects / (len(prediction) + 0.)
      recall = corrects / (len(tags) + 0.)
      if precision == 0 or recall == 0:
          f1Score = 0.0
      else:
        f1Score = 2*precision*recall/(precision + recall)  
    return (f1Score, recall, precision)
  
def predict_tag(data, fileName):
  predicted_tag, distribution = process_data(data)
  distribution_lines = plot_distribution(distribution, fileName+"_tfidf_distribution.svg", format="svg")
  #print(distribution)
  with open('testing1','w') as f:
    print("\n*****************************************\n", file=f);
    print("\n*************" + fileName + "**************\n", file=f);
    print("\n*****************************************\n", file=f);
    for item in distribution:
      print( item, file=f);

  #  print("\n*****************************************\n", file=f);

  #  for item in predicted_tag:
  #   print( item, file=f);

  pd.DataFrame(distribution).to_csv("test.csv")
  output = []
  ylimit = distribution_lines[0].axes.get_ylim()
  max_tfidf = math.ceil((ylimit[0]+ylimit[1])/2)
  mean_tfidf = np.mean(distribution)
  median_tfidf = np.median(distribution)
  print (median_tfidf)
  accuracy_by_id = []
  scores = []
  for i in range(0,len(data)):
    prediction = sorted(predicted_tag[i], key=predicted_tag[i].get, reverse=True)[0: max_tfidf]
    
    #for word in predicted_tag[i]:
      #if(predicted_tag[i][word] >= mean_tfidf):
        #prediction.append(word)
    #prediction = 
    id = data.id[i]
    title = data.title[i]
    content = data.content[i]
    if 'tags' in data:
      tags = data.tags[i]#.replace('-', ' ')
      tagsClean  = set([wordnet_lemmatizer.lemmatize(word) for word in tags.replace('-',' ').split()])
      f1_score = getFScore(prediction, tagsClean)
      scores.append(f1_score[0])
      
      accuracy_by_id.append((id, f1_score[0]))

      tags_from_content = [word for word in tagsClean if word in (title + content).lower()]

      f1_score_content = getFScore(prediction, tags_from_content)           
      
      total_tags = len(tagsClean)
      tags_in_title = sum(1 for word in tagsClean if word in title.lower())
      tags_in_content = sum(1 for word in tagsClean if word in content.lower())
      tags_in_combined = len(tags_from_content)

      count_prediction_from_content_tag = sum(1 for tag in prediction if tag in tags_from_content)
      per_prediction_from_content_tag = 100

      if count_prediction_from_content_tag > 0:
        per_prediction_from_content_tag = (count_prediction_from_content_tag/tags_in_combined)*100
      
      per_tags_in_title = (tags_in_title/total_tags)*100
      per_tags_in_content = (tags_in_content/total_tags)*100
      per_tags_in_combined = (tags_in_combined/total_tags)*100

      output.append([id, ' '.join(prediction), tags, per_prediction_from_content_tag, f1_score[0], f1_score[1], f1_score[2],f1_score_content[0], f1_score_content[1], f1_score_content[2], total_tags, tags_in_title, tags_in_content, tags_in_combined, per_tags_in_title, per_tags_in_content, per_tags_in_combined])      
    else:                  
      #print("Test")      
      output.append([id, title, content, ' '.join(prediction)])
      
  output_file = os.getcwd().replace('script','output')+"/" + fileName + '_output.csv'

  if 'tags' in data:
    print(fileName)
    print(np.average(scores))
    #plot_distribution(scores, fileName+"_accuracy.svg", format="svg")

    columns = ['id','predicted_tags', 'tags', "% predicted tags from content", 'f1_score','recall','precision','f1_score content','recall content','precision content', 'total tags', 'tags in title', 'tags in content', 'tags in combined', '% tags in title', '% tags in content', '% tags in combined']    
    pd.DataFrame(data=output, columns = columns).to_csv(output_file, index=False)  
  else:
    pd.DataFrame(data=output,columns = ['id','title', 'content', 'tags']).to_csv(output_file, index=False)
      
  
def start_processing():
  #print(os.getcwd())
  rawDataDirectory = os.getcwd().replace('script','input'  )
  #print(rawDataDirectory)
  for root,dirs,files in os.walk(rawDataDirectory):    
    for file in files:
      if '.csv' not in file:
        continue

      data = pd.read_csv(os.path.realpath(file).replace('script','input'), encoding = 'latin-1')
      predict_tag(data,os.path.splitext(os.path.basename(file))[0])
  
start_processing()
print("End")