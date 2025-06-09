
import pandas as pd

from collections import Counter
import nltk

from itertools import chain

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import string, re

from nltk import pos_tag
from nltk import ngrams, word_tokenize


import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


from typing import List, Any


df=pd.read_csv(r"file.csv", encoding='Latin-1')

df.loc[:, 'title2'] = df['Title'].str.lower().str.strip().str.rstrip()

df.loc[:, 'text3'] = df['text2'].replace('','')

df['text3'] = df['text3'].astype(str)

df['text4'] = [doc.replace("_", " ") for doc in df['text3']]


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  

df_words = list(sent_to_words(df['title4']))
 
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

df_words_nostops = remove_stopwords(df_words)

def replace_words_in_list(df_words_nostops):
    
    words_to_remove= []
    
    temp = df_words_nostops[:]
    
    for word in words_to_remove:
        if word in temp:
            temp = [w for w in temp if w != word]       
    return temp

newdf_words = [] 

for sublist in df_words:
    result  =replace_words_in_list(sublist)
    newdf_words.append(result)
   
df['new_words']=newdf_words

def remove_punctuation_and_tokenize(word_list):
    text_no_punctuation = [word.translate(str.maketrans('', '', string.punctuation)) for word in word_list]
    tokens = [word_tokenize(word) for word in text_no_punctuation]
    return tokens

def filter_nouns(tokens):
    pos_tags = pos_tag([token for sublist in tokens for token in sublist])
    nouns = [word for word, pos in pos_tags if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
    return nouns

df.loc[:, 'Tokens'] = df['new_words'].apply(remove_punctuation_and_tokenize).apply(filter_nouns)


def calculate_ngrams(tokens, n):
    n_grams = list(ngrams(tokens, n))
    return n_grams

n = 1

df.loc[:,f'Ngrams_{n}'] = df['Tokens'].apply(lambda tokens: calculate_ngrams(tokens, n))


def count_ngram_freq_per_group(group: Any, ngram_column: str, newspaper_column: str) -> Counter:
    
    ngram_list = []
    for ngrams, newspaper in zip(group[ngram_column], group[newspaper_column]):
        for ngram in ngrams:
            ngram_list.append((tuple(map(str, ngram)), newspaper))

    ngram_freq = Counter(ngram_list)

    return pd.DataFrame(ngram_freq.items(), columns=['Ngram_Newspaper', 'Frequency']).assign(
        Ngram=lambda df: df['Ngram_Newspaper'].apply(lambda x: x[0]),
        Newspaper=lambda df: df['Ngram_Newspaper'].apply(lambda x: x[1])
    ).drop(columns=['Ngram_Newspaper'])


ngram_column = 'Ngrams_1'
newspaper_column = 'Newspapers'

df_reset = df.reset_index(drop=True)

ngram_freq_per_year = df_reset.groupby('Year').apply(lambda group: count_ngram_freq_per_group(group, ngram_column,newspaper_column))

ngram_freq_df = ngram_freq_per_year.reset_index(level=0).reset_index(drop=True)

ngram_freq_df['Rank'] = ngram_freq_df.groupby('Year')['Frequency'].rank(method='first', ascending=False)

ngram_freq_df = ngram_freq_df.sort_values(['Year', 'Rank'])


top_ngrams_df = ngram_freq_df[ngram_freq_df['Rank'] <= 100]

top_ngrams_df = top_ngrams_df.sort_values(['Year', 'Newspaper','Rank'])



def clean_ngram(ngram):
    if isinstance(ngram, tuple):
        ngram_str = ' '.join(ngram)  # Join tuple elements with space
    elif isinstance(ngram, str):
        ngram_str = ngram
    else:
        return ''  

    return re.sub(f'[{string.punctuation}]', '', ngram_str)

ngram_freq_df['Ngram'] = ngram_freq_df['Ngram'].astype(str).fillna('')
ngram_freq_df['Ngram'] = ngram_freq_df['Ngram'].apply(clean_ngram)



