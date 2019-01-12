# -*- coding: utf-8 -*-
###################### OLD VERSION ######################

""" 
tokenizing wiki-corpus 
input: text_files
output: <list>words, <list> words_id
"""

import deepcut
import json
import numpy as np
import os
import re
import unicodedata

np.random.seed(0)

### path to datasets
path = 'C:\\Users\\Patdanai\\Desktop\\wiki-dictionary-[1-50000]'
datasets = os.listdir(path)

### random training file
samples = []
for i in range(100):
    random_i = np.random.randint(len(datasets))
    samples.append(int(datasets[random_i].split('.')[0]))
samples = sorted(samples)

### get randomed document ids
def get_doc_id(data):
    tmp = ''.join(data[0:5])
    doc_id = re.findall('\d+', tmp)
    return int(doc_id[0])

# get words/vocabularies from file
x = []
i = 1
for f in samples:
    dataset_path = os.path.join(path, str(f) + '.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for j in range(len(data)):
            if(data[j] == ''):
                data[j] = data[j].replace('', ' ') # remove double quotes from data
    x.append(data)
    i += 1

dictionary = [w for i in list(x) for w in i]
w2id = {}
w2id = {w: ind if not(w.isdigit()) else w for ind, w in enumerate(set(dictionary))}
values = dictionary

# for (i, w) in enumerate(set(dictionary)):
#     try:
#         if(not(w.isdigit())):
#             w2id[w] = i
#         else:
#             w2id[int(w)] = i
#     except ValueError:
#         pass

id2w = {idn: w for w, idn in w2id.items()}
samples_ids = np.asarray(samples).reshape((len(samples), ))
y = doc_ids = samples_ids

# random input, output index of one sample simultaneously
temp = list(zip(x, y))
np.random.shuffle(temp)
x, y = zip(*temp)

id_representation = []
for doc in x:
    id_representation.append([w2id[w] for w in doc])
# print(id_representation[0], y[0])

word_and_doc_id = []
for i in range(len(id_representation)):
    # word_and_doc_id += list(zip(id_representation[i], [y[i] for j in range(len(id_representation[i]))]))
    word_and_doc_id.append(id_representation[i])

# print function
# last_doc_id = max(y)
# for i in range(len(word_and_doc_id)):
#     print(word_and_doc_id[i], y[i])

doc_ids = np.asarray(y).reshape((len(doc_ids), ))

from keras.utils import to_categorical
doc_ids = to_categorical(doc_ids) # to classes
# print(doc_ids[0][42827])

# avrage_word_per_doc = sum([len(i) for i in id_representation]) / len(id_representation)
# print(int(avrage_word_per_doc))

max_word_per_doc = max([len(i) for i in id_representation])
# print(max_word_per_doc)

#### 20-words sentences
n = 20
sentences = word_and_doc_id.copy()
for i in range(len(sentences)):
    sentences[i] = [sentences[i][j * n:(j + 1) * n] for j in range((len(sentences[i]) + n - 1) // n )] 
####

# pad sequence TODO sentence to id
from keras.preprocessing import sequence
n_words_sentences = []
for i in range(len(sentences)):
    try:
        n_words_sentences += list(zip(sequence.pad_sequences(sentences[i], maxlen=n), 
                                [y[i] for j in range(len(sentences[i]))]))
    except ValueError:
        pass

sl, d_ids = zip(*n_words_sentences)
# print(sl, '-----')
# print(sl, '*****')

s_train = np.asarray(sl[:int(.8 * len(sl))])
s_test = np.asarray(sl[int(.8 * len(sl)):])
print(s_train, s_train.shape)

doc_id_train = np.asarray(d_ids[:int(.8 * len(d_ids))])
doc_id_test = np.asarray(d_ids[int(.8 * len(d_ids)):])
print(doc_id_train, doc_id_train.shape)

# print(s_train[0])

# building model
from keras.models import Model
from keras import Sequential
from keras.layers import Activation, Bidirectional, Embedding, LSTM, Dense, Dropout, TimeDistributed, InputLayer
embedding_size=64
lstm_output = 128
model = Sequential()
model.add(InputLayer(input_shape=s_train.shape)) # 3164*20
model.add(Embedding(len(id2w), embedding_size)) 
print(model.summary())
model.add(Bidirectional(LSTM(lstm_output, return_sequences=True)))
model.add(TimeDistributed(Dense(max(y) + 1)))
model.add(Activation('softmax'))
# model.add(LSTM(lstm_output))
# model.add(TimeDistributed(Dense(max(y) + 1, activation='softmax',), input_shape=(20, )))
# drop out regularization
print(model.summary())
print()

# training and evaluation
model.compile(loss='categorical_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

batch_size = 32
epochs = 4
s_valid, doc_id_valid = s_train[:batch_size], doc_id_train[:batch_size]
s_train2, doc_id_train2 = s_train[batch_size:], doc_id_train[batch_size:]
# print(s_train2)
# print(doc_id_valid)
model.fit(s_train2, doc_id_train2, validation_data=(s_valid, doc_id_valid), batch_size=batch_size, epochs=epochs)
scores = model.evaluate(s_test, doc_id_test, verbose=0)
print('Test accuracy:', scores[1])
print()

# print(s_test)
# print(model.predict(s_test))
print()

# save model
