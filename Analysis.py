#Word cloud
import pandas as pd
from os import path
import glob
import json
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
import re


def readTextsIntoDf(filepath=""):
	if filepath == "":
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
	else:
		dataDict = {}
		with open(filepath, "r") as f:
			datas = []
			line = f.read().strip("[").strip("]").strip("\n")
			datas.extend([x for x in line.split(",")])
		#print(datas[:3])


		dataDf = pd.DataFrame(datas)
		dataDf.rename(columns={0: "content"}, inplace=True)
		print ("Data is successfully stored in df ...")
		#print (dataDf)
		
	return dataDf
	
df = readTextsIntoDf("processed.txt")


#df = pd.read_csv("processed.txt")

processedData = df.content.values.tolist()
corpus = []
for word in processedData:
	if "_" in word:
		word = word.replace("_", " ")
	corpus.append(word)




#Identify common words
freq = pd.Series(' '.join(df['content']).split()).value_counts()[:20]
print (freq)

#Identify uncommon words
freq1 =  pd.Series(' '.join(df 
         ['content']).split()).value_counts()[-20:]
print (freq1)


wordcloud = WordCloud(
                          background_color='white',
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(processedData))
#print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#fig.savefig("word1.png", dpi=900)
cv = CountVectorizer(max_df = 0.8, max_features=10000, ngram_range=(1, 3))
X = cv.fit_transform(processedData)

print (list(cv.vocabulary_.keys())[:10])

#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus,30)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
print(top_df)
#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()
	

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
top2_words = get_top_n2_words(corpus, n=30)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)
#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)
plt.show()
