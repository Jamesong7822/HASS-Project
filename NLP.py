import requests
from bs4 import BeautifulSoup
import pickle
import re
import numpy as np
import pandas as pd
from pprint import pprint

import os
import time
import json
import glob

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy


from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 10000000 # increase size





# Get Links
#homeLink = "https://dailysciencefiction.com"
homeLink = "https://www.lightspeedmagazine.com/category/fiction/science-fiction/page/"

		  
def genMonths(year1=2010, year2=2020):
	monthsList = []
	for year in range(year1, year2):
		if year == 2010:
			for month in ["09","10","11","12"]:
				toAppend = str(year) + "." + month
				monthsList.append(toAppend)
		else:
			for month in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
				toAppend = str(year) + "." + month
				monthsList.append(toAppend)
				
	return monthsList
	
def genPageUrls(num1, num2):
	pageNums = []
	for num in range(num1, num2+1):
		pageNums.append(homeLink + str(num))
	return pageNums

def getPageLinks(url, month):
	page = requests.get(url).text
	soup = BeautifulSoup(page, "lxml")
	allLinks = []
	for data in soup.findAll("div", attrs={"class":"post_content"}):
		for title in data.findAll("h2", attrs={"class":"posttitle entry-title"}):
			links = title.findAll("a")
			for link in links:
				allLinks.append(link["href"])
	return allLinks
			
def writeFile(fileName, content, mode="w+"):
	with open(fileName, mode) as f:
		#json.dump(content, f)
		for i in content:
			f.write(str(i))
			f.write("/n")
		#pickle.dump(content, f)
	

def getAllPageLinks(monthsList):
	allLinks = {}
	for month in monthsList:
		print ("Retrieving url links for time of: {} ...".format(month))
		url = homeLink + "/month/stories/" + month
		content = getPageLinks(url, month)
		year = month[:4]
		if year not in allLinks:
			allLinks[year] = content
		else:
			allLinks[year].extend(content)
		
		fileName = "urls.text"
	print ("Writing Urls Data into file: {}".format(fileName))
	writeFile(fileName, allLinks, "w+")	


"""
urls = ["http://www.lightspeedmagazine.com/fiction/harry-and-marlowe-and-the-talisman-of-the-cult-of-egil/",
		"http://www.lightspeedmagazine.com/fiction/harry-and-marlowe-escape-the-mechanical-siege-of-paris/",
		"http://www.lightspeedmagazine.com/fiction/harry-and-marlowe-and-the-intrigues-at-the-aetherian-exhibition/",
		"http://www.lightspeedmagazine.com/fiction/harry-and-marlowe-and-the-secret-of-ahomana/",
		"http://www.lightspeedmagazine.com/fiction/marlowe-and-harry-and-the-disinclined-laboratory/",
		"http://www.lightspeedmagazine.com/fiction/the-path-of-pins-the-path-of-needles/",
		"http://www.lightspeedmagazine.com/fiction/today-is-today/",
		"http://www.lightspeedmagazine.com/fiction/eros-pratfalled-or-adrift-in-the-cosmos-with-lasagna-and-mary-steenburgen/",
		]
		
urls1 = ["http://strangehorizons.com/fiction/",
		"http://strangehorizons.com/fiction/2086/",
		"http://strangehorizons.com/fiction/the-fortunate-death-of-jonathan-sandelson/",
		"http://strangehorizons.com/fiction/directions/",
		"http://strangehorizons.com/fiction/the-palace-of-the-silver-dragon/",
		"http://strangehorizons.com/fiction/copy-cat/",
		"http://strangehorizons.com/fiction/we-feed-the-bears-of-fire-and-ice/",
		"http://strangehorizons.com/fiction/the-glow-in-the-dark-girls/",
		"http://strangehorizons.com/fiction/them-boys/",
		"http://strangehorizons.com/fiction/the-gardens-first-rule/",
		"http://strangehorizons.com/fiction/whom-my-soul-loves/",
		"http://strangehorizons.com/fiction/promise-me-this-is-ours/",
		"http://strangehorizons.com/fiction/the-darwinist-%d8%a7%d9%84%d8%af%d8%a7%d8%b1%d9%88%d9%8a%d9%86%d9%8a/",
		"http://strangehorizons.com/fiction/seed-vault/",
		"http://strangehorizons.com/fiction/spider/",
		"http://strangehorizons.com/fiction/invisible-and-dreadful/",
		"http://strangehorizons.com/fiction/many-hearted-dog-and-heron-who-stepped-past-time/",
		"http://strangehorizons.com/fiction/gephyrophobia/",
		"http://strangehorizons.com/fiction/the-kings-mirror/"
		]
"""




def url_to_transcript(url):
	page = requests.get(url).text
	soup = BeautifulSoup(page,"lxml")
	data = soup.findAll("div", attrs={"class": "entry clear single_entry entry-content"})
	allTexts = []
	for d in data:
		for p in d.findAll("p"):
			allTexts.append(p.text)
			#allTexts.append([a.text for a in d.findAll("p")])
	combined = " ".join(allTexts)
	#text = [p.text for p in soup.findAll("storyText").find_all('p')]
	#print(url)
	#return combined
	#print (combined)
	return combined
	
def downloadTexts(data, counter=0):
	fileName = os.path.join(os.path.normpath(os.getcwd()) + "/Articles/rawdata.txt")
	
	relevantData = data["http"]
	for url in relevantData:
		print ("Retrieving text from url: {}".format(url))
		text = url_to_transcript(url)
		text = text.encode('ascii',errors='ignore').decode()
		#print (text)
		writeFile(fileName, text, "a+")
		counter += 1
			
	return counter

def downloadTextsByYear(data, year, counter=0):
	fileName = os.path.join(os.path.normpath(os.getcwd()) + "/Articles/" + str(year) + ".txt")
	try:
		relevantData = data[str(year)]
		#print (len(relevantData))
		
		for url in relevantData:
			print ("Retrieving text from url: {}".format(url))
			text = url_to_transcript(url)
			text = text.encode('ascii',errors='ignore').decode()
			#print (text)
			writeFile(fileName, text, "a+")
			counter += 1
	except:
		print ("Error: {} is not a key in the data dict".format(year))
		return counter
			
	return counter
	
def readTextsIntoDf():
	dataPaths = "Articles/"
	filepaths = glob.glob(dataPaths+"*.txt")
	#print (filepaths)
	dataDict = {}
	for index, filepath in enumerate(filepaths):
		print ("Reading: {}".format(filepath))
		with open(filepath, "r+") as f:
			datas = []
			for line in f:
				datas.append(line)
				
		combinedData = " ".join(datas)
		dataDict[index] = combinedData
		
	dataDf = pd.DataFrame(dataDict, index=[0]).transpose()
	dataDf.rename(columns={0: "content"}, inplace=True)
	print ("Data is successfully stored in df ...")
	print (dataDf)
	return dataDf
	
def sent_to_words(sentences):
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
		
def remove_stopwords(texts, stop_words):
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
	return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, trigram_mod):
	return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
	texts_out = []
	for sent in texts:
		doc = nlp(" ".join(sent)) 
		texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
	return texts_out
	
def processData(df, stop_words):
	print ("Processing Data ...")
	data = df.content.values.tolist()
	data_words = list(sent_to_words(data))
	
	# Build the bigram and trigram models
	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
	
	# Faster way to get a sentence clubbed as a trigram/bigram
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)

	# Define functions for stopwords, bigrams, trigrams and lemmatization
	
	
	# Remove Stop Words
	data_words_nostops = remove_stopwords(data_words, stop_words)

	# Form Bigrams
	data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

	
	
	# Do lemmatization keeping only noun, adj, vb, adv
	data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'ADV'])

	#print(data_lemmatized[:])
	print ("Processing Complete!")
	processedWordsFilepath = "processed.txt"
	print ("Processed words in {}".format(processedWordsFilepath))
	writeFile(processedWordsFilepath, data_lemmatized, "w+")
	return data_lemmatized

def getTopics(data, numTopics=10, savePath="lda.model"):
	print ("Preparing Model")
	# Create Dictionary
	id2word = corpora.Dictionary(data)

	# Create Corpus
	texts = data

	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in texts]

	# View
	#print(corpus[:1])

	# Build LDA model
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
											   id2word=id2word,
											   num_topics=numTopics, 
											   random_state=100,
											   update_every=10,
											   chunksize=100,
											   passes=50,
											   alpha='auto',
											   per_word_topics=True,
											  )
												  
		   
	# Print the Keyword in the 10 topics
	#pprint(lda_model.print_topics(num_words=10))
	doc_lda = lda_model[corpus]
	
	#coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
	#coherence_lda = coherence_model_lda.get_coherence()
	#print ("{} topics gave a coherence score of {}".format(numTopics, coherence_lda))
	
	# Save the model
	#bestModel.save(savePath)
	#print ("Best Coherence Score Model saved to {}".format(savePath))
	#return coherence_lda, model
	return lda_model, id2word
		
		
			
def run():
	startTime = time.time()
	counter = 0
	urlsFileName = "urls.text"
	if not os.path.exists(urlsFileName):
		print ("Urls text file does not exist. Retrieving Now..")
		#monthsList = genMonths()
		pageNums = genPageUrls(1, 50)
		print ("Retrieving Urls Now...")
		getAllPageLinks(pageNums)
		
		# Urls datafile exists!
		print ("Found Urls Data File!")
		with open(urlsFileName, "r") as f:
			#data = pickle.load(f)
			data = json.load(f)
		#for year in [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
		#	counter = downloadTextsByYear(data, year)
		counter = downloadTexts(data, counter)
			
		elapsedTime = time.time() - startTime
		hours = elapsedTime // 3600
		temp = elapsedTime - 3600 * hours
		minutes = temp // 60
		seconds = temp -60 * minutes
		print ("Downloaded {} articles. Elapsed Time: {} hours {} minutes {} seconds".format(counter, hours, minutes, seconds))
			
	elif os.path.exists(urlsFileName):
		
		print ("Found Urls Data File!")
		# Check if Articles exists
		def checkData():
			articles = "/Articles/"
			filepath = os.getcwd() + articles + "rawdata.txt"
			exists = os.path.exists(filepath)
			
			return exists
		exists = checkData()
		if not exists:
			print ("Data files not found. Preparing to download now ...")
		
			with open(urlsFileName, "r") as f:
				#data = pickle.load(f)
				data = json.load(f)
				#print (data)
			#for year in [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]:
			#	counter = downloadTextsByYear(data, year, counter)
			counter = downloadTexts(data, counter)	
			
			elapsedTime = time.time() - startTime
			hours = elapsedTime // 3600
			temp = elapsedTime - 3600 * hours
			minutes = temp // 60
			seconds = temp -60 * minutes
			print ("Downloaded {} articles. Elapsed Time: {} hours {} minutes {} seconds".format(counter, hours, minutes, seconds))
			
		else:
			print ("Data files present")
			print ("Reading them now ...")
			
			bestScore = 0
			bestModel = None
			dataDf = readTextsIntoDf()
			processedData = processData(dataDf, stop_words)
			#model, id2word = getTopics(processedData)
			
			
			for numTopics in range(3, 25):
				model, id2word = getTopics(processedData, numTopics)
				coherence_model_lda = CoherenceModel(model=model, texts=processedData, dictionary=id2word, coherence='c_v')
				score = coherence_model_lda.get_coherence()
				print ("Topics: {} Coherence Score: {}".format(numTopics, score))
				
				if score > bestScore:
					bestScore = score
					bestModel = model
				else:
					del model
					
			pprint(bestModel.print_topics(num_words=10))
			# save the best model
			bestModel.save("bestModel.model")
			# save the output 
			writeFile("Output.txt", str(bestModel.print_topics(num_words=10)), "w+")
			#pprint(bestModel.print_topics(num_words=10))
			
			elapsedTime = time.time() - startTime
			hours = elapsedTime // 3600
			temp = elapsedTime - 3600 * hours
			minutes = temp // 60
			seconds = temp - 60 * minutes
			print ("Finished Modellling. Elapsed Time: {} hours {} minutes {} seconds".format(hours, minutes, seconds))
			
	else:
		pass
			

		
		
if __name__ == "__main__":
	# Load stopwords
	extraStopWords = []
	print ("Loading extra stop words!")
	with open("stopwords.txt", "r") as f:
		for word in f:
			extraStopWords.append(word.strip("\n"))
	stop_words.extend(extraStopWords)
	
	stopWords2 = ["time"]
	stop_words.extend(stopWords2)
	print ("Stop words updated!")
	#print (stop_words)
	run()

# with open("Articles/2011.txt", "rb") as f:
	# while True:
		# try:
			# data = pickle.load(f).decode("utf-8")
			# print (data)
		# except:
			# break