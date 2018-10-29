# -*- coding: utf-8 -*-

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
# print(datasets)

### random training file
samples = []
for i in range(100):
    random_i = np.random.randint(len(datasets))
    samples.append(int(datasets[random_i].split('.')[0]))

samples = sorted(samples)
# print(samples)

### get randomed document ids
def get_doc_id(data):
    tmp = ''.join(data[0:5])
    doc_id = re.findall('\d+', tmp)
    return int(doc_id[0])

### group words to n-words sentences
def n_word_sentence_segment(n, data):
    data = [data[i * n:(i + 1) * n] for i in range((len(data) + n - 1) // n )] 
    for i in range(len(data)):
        data[i] = ''.join(data[i]) # merge words
    return data

x = []
i = 1
for f in samples:
    dataset_path = os.path.join(path, str(f) + '.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print('1231', data)

        for j in range(len(data)):
            data[j] = data[j].replace('"', '').strip(' ')
            if(data[j] == ''):
                data[j] = data[j].replace('', ' ') # remove double quotes from data
    # print('*****', i, (data))
    doc_id = get_doc_id(data)
    data = n_word_sentence_segment(20, data)
    x.append(data)
    # print(str(i) + ' -', doc_id)
    i += 1

samples_ids = np.asarray(samples).reshape((len(samples), 1))
y = doc_ids = samples_ids
print(len(x))
print(len(y))

temp = list(zip(x, y))
np.random.shuffle(temp)
x, y = zip(*temp)

# print(x[99], y[99])

# json.dump(data, open('test.txt', 'w', encoding='utf-8-sig'), ensure_ascii=True) # save sentence
# for i in range(len(y)):
#     for j in range(len(data)):
#         print(i, [data[j], y[i]])

# print(data)

sentences = list(x)
all_sentences = [s for i in list(sentences) for s in i]
s2id = {s: idn if not(s.isdigit()) else int(s) for idn, s in enumerate(set(all_sentences))}
id2s = {idn: s for s, idn in s2id.items()}

id_representation_sentences = []
for s in sentences:
    id_representation_sentences.append([s2id[i] for i in s])
doc_ids = y
n_sentences = max([len(i) for i in sentences])
print(n_sentences)

s_train = id_representation_sentences[:int(.8 * len(sentences))]
s_test = id_representation_sentences[int(.8 * len(sentences)):]
id_train = doc_ids[:int(.8 * len(sentences))]
id_test = doc_ids[int(.8 * len(sentences)):]

# word database
# word2id = imdb.get_word_index() # value to key

try:
    id2word = {i: word for word, i in word2id.items()} # key to value
    print('---review with words-id---')
    print(X_train[0]) # example of x_train
    print('---review with words---')
    print([id2word.get(i, ' ') for i in X_train[0]]) # convert ids to words in x_train
    print('---label---')
    print(y_train[0]) # example of y_train
    print()

    print('Minimum review length: {}'.format(len(min((X_test + X_test), key=len))))
    print('Maximum review length: {}'.format(len(max((X_train + X_test), key=len))))
    print()
except:
    pass

# pad sequence TODO sentence to id
from keras.preprocessing import sequence

# max_words = 500 # to have same shape in each sample by padding below
# print('--------------------tr--------------------', s_train[0])
# print('--------------------te--------------------', s_test[0])
s_train = sequence.pad_sequences(s_train, maxlen=n_sentences)
s_test = sequence.pad_sequences(s_test, maxlen=n_sentences)
print('---example review with padded words-id---')
print(s_train[0])
print()
exit()

# building model
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
embedding_size=64
lstm_output = 192
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(lstm_output))
model.add(Dense(len(doc_ids), activation='softmax'))
# drop out regularization
print(model.summary())
print()

# training and evaluation
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

batch_size = 64
epochs = 8
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs)
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
print()

# save model 
