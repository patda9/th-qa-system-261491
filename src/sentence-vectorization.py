import json
import numpy as np
import os
import preprocessing as prep
import re

np.random.seed(0)

path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/'
dataset = os.listdir(path)
# print(dataset.__len__())

n = 100
sample_ids = []
for i in range(n):
    randomed_doc_id = int(dataset[np.random.randint(dataset.__len__())].split('.')[0])
    while(randomed_doc_id in sample_ids):
        randomed_doc_id = int(dataset[np.random.randint(dataset.__len__())].split('.')[0])
    sample_ids.append(randomed_doc_id)

# print(sample_ids, sample_ids.__len__())

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
# print(samples[-1])
vocabularies = [w for doc in samples for w in doc]
# print(vocabularies[-1])
vocabularies = prep.remove_stop_words(vocabularies)
# print(vocabularies[-1])

for i in range(samples.__len__()):
    samples[i] = prep.remove_xml(samples[i])
    samples[i] = prep.remove_stop_words(samples[i])

## word to id transformation
word2id = {}
word2id['<PAD>'] = 0
word2id['<NIV>'] = 1
for (i, w) in enumerate(set(vocabularies), 2):
    try:
        word2id[w] = i
    except ValueError:
        word2id[w] = 1

id2word = {idx: w for w, idx in word2id.items()}

# print(id2word)

sample2id = {article_id: i for i, article_id in enumerate(list(sample_ids))}
# print(sample2id)

## words to word ids representation
id_representation = []
for i in range(samples.__len__()):
    id_representation.append([word2id[w] for w in samples[i]])

# print(id_representation)

remapped_article_ids = []
for i in range(sample_ids.__len__()):
    remapped_article_ids.append(sample2id[sample_ids[i]])

# print(remapped_article_ids)

words_per_sentence = 20
article_sentences = id_representation.copy()
for i in range(article_sentences.__len__()):
    article_sentences[i] = [article_sentences[i][j*words_per_sentence : (j+1)*words_per_sentence] for j in range((article_sentences[i].__len__() + words_per_sentence-1) // words_per_sentence)]

# print('*', article_sentences[0])

from keras.preprocessing import sequence

padded_article_sentences = []
for i in range(article_sentences.__len__()):
    padded_s = sequence.pad_sequences(article_sentences[i], maxlen=words_per_sentence)
    padded_article_sentences.append(padded_s)

# print(padded_article_sentences)

sentences_article_ids = []
y_train_category = []
for i in range(padded_article_sentences.__len__()):
    y_train_category.append(remapped_article_ids[i])
    for j in range(padded_article_sentences[i].__len__()):
        sentences_article_ids.append(remapped_article_ids[i])
# print(sentences_article_ids)

flatten_article_sentences = np.vstack(padded_article_sentences)
# print(list(zip(flatten_article_sentences, sentences_article_ids)))

from keras.utils import to_categorical

x = flatten_article_sentences.copy()
y = sentences_article_ids.copy()
input_output_label = list(zip(x, y))
np.random.shuffle(input_output_label)
x, y = list(zip(*input_output_label))
x = np.asarray(list(x))
y = np.asarray(list(y))

x_train = x[:int(.8 * x.__len__())]
y_train = y[:int(.8 * y.__len__())]
x_test = x[int(.8 * x.__len__()):]
y_test = y[int(.8 * y.__len__()):]

from keras.layers import Activation, Bidirectional, Dense, Embedding, Flatten, InputLayer, LSTM, TimeDistributed
from keras.layers.core import Masking
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

embedding_size = 64
lstm_output_size = 128

model = Sequential()
model.add(Embedding(word2id.__len__(), embedding_size, input_length=words_per_sentence))
model.add(Masking(mask_value=0, input_shape=(words_per_sentence, 1)))
model.add(Bidirectional(LSTM(lstm_output_size)))
model.add(Dense(64, activation='relu'))
model.add(Dense(sample2id.__len__(), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

model.fit(x_train, to_categorical(y_train, num_classes=sample2id.__len__()), batch_size=32, epochs=8)

# model.save('../sentence-vectorization-model.h5')

scores = model.evaluate(x_test, to_categorical(y_test, num_classes=sample2id.__len__()))
print(scores)
print(f"{model.metrics_names[1]}: {scores[1] * 100}%")

dense1_layer = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
dense1_layer_output = dense1_layer.predict(np.asarray(x_test))
print('sentence vectorization output vvv ( dimension:', dense1_layer_output[0].__len__(), ')')
print(dense1_layer.predict(np.asarray(x_test)))
softmax_layer = Model(inputs=model.input, outputs=model.get_layer(index=4).output)
softmax_layer_output = softmax_layer.predict(np.asarray(x_test))
print('softmax classification probabilities output vvv ( classes:', softmax_layer_output[0].__len__(), ')')
print(softmax_layer.predict(np.asarray(x_test)))

prediction = model.predict(x_test)
# print(prediction)
