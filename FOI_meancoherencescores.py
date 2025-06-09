
import pandas as pd

import re

import numpy as np

import os

import nltk

import pyLDAvis.gensim

import spacy


from pprint import pprint

import seaborn as sns

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

import pyLDAvis

import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook() 

from wordcloud import WordCloud

import glob

import matplotlib.pyplot as plt 
import matplotlib.ticker

from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import numpy as np

plt.style.use('ggplot')


plt.rcParams['figure.figsize'] = (8,6) #can change the size of bars

plt.rcParams['font.size'] = 12 #can change the size of font

from nltk.corpus import stopwords

stop_words = stopwords.words('english')


import os
from gensim.models import LdaModel

os.environ['MALLET_HOME'] = 'E:\\temp\\mallet-2.0.8'

df=pd.read_csv(r"file.csv", encoding='Latin-1')

df.loc[:, 'text2'] = df['Text'].str.lower().str.strip().str.rstrip()
df['text2'].head()

df.loc[:, 'text3'] = df['text2'].replace('','')

df['text3'] = df['text3'].astype(str)

df['text4'] = [doc.replace("_", " ") for doc in df['text3']]


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  

df_words = list(sent_to_words(df['text4']))
                 

words_to_remove=[]

def remove_words_from_list(df_words, words_to_remove):
    
    return [word for word in df_words if word not in words_to_remove]

newdf_words = [remove_words_from_list(sublist, words_to_remove) for sublist in df_words]

bigram = gensim.models.Phrases(newdf_words, min_count=5, threshold=100) 

trigram = gensim.models.Phrases(bigram[newdf_words], threshold=100)  


bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

df_words_nostops = remove_stopwords(newdf_words)


df_words_bigrams = make_bigrams(df_words_nostops)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


df_lemmatized = lemmatization(df_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


df['df_lemmatized'] = df_lemmatized


id2word = corpora.Dictionary(df_lemmatized)

texts = df_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]




def compute_mean_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    
    coherence_values = {}
    model_list = []

    for num_topics in range(start, limit, step):
        topic_coherence_values = []
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        topic_coherence_values.append(coherencemodel.get_coherence())
        
        if num_topics not in coherence_values:
            coherence_values[num_topics] = []
        coherence_values[num_topics].append(coherencemodel.get_coherence())

    mean_coherence_values = {num_topics: np.mean(values) for num_topics, values in coherence_values.items()}
    return model_list, mean_coherence_values


model_list, mean_coherence_values = compute_mean_coherence_values(dictionary=id2word, corpus=corpus, texts=df_lemmatized, start=2, limit=40, step=3)
print("Mean Coherence Values: ", mean_coherence_values)


topics = list(mean_coherence_values.keys())
coherence = list(mean_coherence_values.values())

plt.figure(figsize=(10, 5))
plt.plot(topics, coherence, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Mean Coherence Score')
plt.title('Mean Coherence Score by Number of Topics')
plt.grid(True)
plt.savefig('')
plt.show()