import json
import numpy as np
import os
import preprocessing as prep
import re

np.random.seed(0)

path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/'
dataset = os.listdir(path)
print(dataset.__len__())

n = 100
sample_ids = []
for i in range(n):
    randomed_doc_id = int(dataset[np.random.randint(dataset.__len__())].split('.')[0])
    while(randomed_doc_id in sample_ids):
        randomed_doc_id = int(dataset[np.random.randint(dataset.__len__())].split('.')[0])
    sample_ids.append(randomed_doc_id)

print(sample_ids, sample_ids.__len__())

count = 1
samples = []
for article_id in sample_ids:
    text_file_path = os.path.join(path + str(article_id) + '.json')
    with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as text_file:
        data = json.load(text_file)
    samples.append(data)
    count += 1

# print(samples[0])
# print(sample_ids[0])

for i in range(samples.__len__()):
    samples[i] = prep.remove_xml(samples[i])
print(samples[-1])
vocabularies = [w for doc in samples for w in doc]
print(vocabularies[-1])
vocabularies = prep.remove_stop_words(vocabularies)
print(vocabularies[-1])

for i in range(samples.__len__()):
    samples[i] = prep.remove_xml(samples[i])
    samples[i] = prep.remove_stop_words(samples[i])

## word to id transformation
word2id = {}
word2id['<NIV>'] = 0
for (i, w) in enumerate(set(vocabularies), 1):
    try:
        word2id[w] = i
    except ValueError:
        word2id[w] = 0

id2word = {idx: w for w, idx in word2id.items()}

print(id2word)

sample2id = {i: article_id for i, article_id in enumerate(list(sample_ids))}
print(sample2id)

## words to word ids representation
id_representation = []
for i in range(samples.__len__()):
    id_representation.append([word2id[w] for w in samples[i]])

print(id_representation)

doc_ids_remapping = []
for i in range(sample_ids.__len__()):
    doc_ids_remapping.append(sample2id[i])

print(doc_ids_remapping)

words_per_sentence = 10
article_sentences = id_representation.copy()
for i in range(article_sentences.__len__()):
    article_sentences[i] = [article_sentences[i][j*words_per_sentence : (j+1)*words_per_sentence] for j in range((article_sentences[i].__len__() + n-1) // n)]

print('*', article_sentences[0])

# https://www.facebook.com/LADbible/videos/375311296563614/

from keras.preprocessing import sequence

padded_article_sentences = []
for i in range(article_sentences.__len__()):
    try:
        padded_s = sequence.pad_sequences(article_sentences[i], maxlen=words_per_sentence)
        padded_article_sentences.append()
    except ValueError:
        pass

from keras.layers import Activation, Bidirectional, Dense, Embedding, Flatten, InputLayer, LSTM, TimeDistributed
from keras.layers.core import Masking
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

embedding_size = 64

model = Sequential()
# model.add(Embedding(word2id.__len(), ))
