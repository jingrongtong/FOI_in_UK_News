
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

plt.style.use('ggplot')


plt.rcParams['figure.figsize'] = (8,6) 

plt.rcParams['font.size'] = 12 

from nltk.corpus import stopwords

stop_words = stopwords.words('english')


import os
from gensim.models import LdaModel

os.environ['MALLET_HOME'] = 'E:\\temp\\mallet-2.0.8'

df=pd.read_csv(r"file.csv", encoding='Latin-1')

df.loc[:, 'text2'] = df['Text'].str.lower().str.strip().str.rstrip()

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

lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=32, random_state=42)

topics = lda.show_topics(num_topics=32, num_words=10)
for topic in topics:
    print(topic)

for topic in topics:
    
    for word in topic[1].split(' + '):
        word = word.split('*')[1].strip().replace('"', '')
        if word not in id2word.token2id:
            print(f"Unexpected word: {word}")


print('\nPerplexity: ', lda.log_perplexity(corpus)) 

coherence_model_lda = CoherenceModel(model=lda, texts=df_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence() 
print('\nCoherence Score: ', coherence_lda)

pyLDAvis.enable_notebook()
vis= pyLDAvis.gensim.prepare(lda, corpus, id2word) 

topic_distribution = [lda.get_document_topics(document, minimum_probability=0) for document in corpus]

topic_distribution_df = pd.DataFrame([{f"Topic_{topic_id}": topic_prob for topic_id, topic_prob in topics} for topics in topic_distribution])

combined_df = pd.concat([df.reset_index(drop=True), topic_distribution_df], axis=1)


numeric_columns = combined_df.select_dtypes(include=[np.float32, np.float64])

topic_distribution_by_document = combined_df.groupby('Index')[numeric_columns.columns].mean().reset_index()

topic_distribution_by_year_publication = combined_df.groupby(['Index','Year', 'Newspapers'])[numeric_columns.columns].mean().reset_index()

topic_distribution_by_year_publication.to_csv('', index=False)


topics = lda.print_topics()

text = ' '.join([topic[1] for topic in topics])


font_path = r''

wordcloud = WordCloud(width=800, height=400, background_color='white',font_path=font_path).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



topics_df = pd.DataFrame(topics, columns=['Topic_Number', 'Words'])

topics_df.to_csv('')

coherence_model_lda_model = CoherenceModel(model=lda, texts=df_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda_model = coherence_model_lda_model.get_coherence()
print('\nCoherence Score: ', coherence_lda_model)


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        
        model = LdaModel(corpus=corpus, id2word=id2word,num_topics=num_topics, random_state=42)
       
        model_list.append(model)
       
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=df_lemmatized, start=2, limit=40, step=3)

limit=40; start=2; step=3;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


target_topic_number = 32  
print("Starting loop...")
for m, cv in zip(x, coherence_values):
    if m == target_topic_number:
        print("Coherence Value for Topic", m, ":", round(cv, 4))

model_list

for idx, model in enumerate(model_list):
    if model.num_topics == lda.num_topics:
        print(f"Index of lda_model in model_list: {idx}")
        break

optimal_model = model_list[10] 
num_topics = len(optimal_model.get_topics())  
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10, num_topics=num_topics))


topics=optimal_model.print_topics(num_words=10,num_topics=num_topics)

topics_df = pd.DataFrame(topics, columns=['Topic_Number', 'Words'])

topics_df.to_csv('')

def format_topics_sentences(ldamodel=lda, corpus=corpus, texts=df_lemmatized):
    
    sent_topics_df = pd.DataFrame()

    
    num_topics = ldamodel.num_topics
    print(f"Number of topics in the model: {num_topics}")
    
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: 
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                

                data = {'Topic_Num': [int(topic_num)], 'Prop_Topic': [round(prop_topic, 4)], 'Topic_Keywords': [topic_keywords]}
                
                df = pd.DataFrame(data)
                
                sent_topics_df = pd.concat([sent_topics_df,df], ignore_index=True)
                
            else:
                break
                 
            
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=df_lemmatized)


df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']


df_dominant_topic.index.name = "index"
df_dominant_topic
