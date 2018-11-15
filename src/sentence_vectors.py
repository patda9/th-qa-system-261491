import deepcut
import json
import numpy as np
import os
import re
import unicodedata

### fixed seed
np.random.seed(0)

### path to datasets
path = 'C:\\Users\\Patdanai\\Desktop\\wiki-dictionary-[1-50000]'
datasets = os.listdir(path)

### random training file
samples = []
for i in range(200):
    random_i = np.random.randint(len(datasets))
    samples.append(int(datasets[random_i].split('.')[0]))
samples = sorted(samples)
print(len(samples))

### get randomed document ids
def get_doc_id(data):
    tmp = ''.join(data[0:5])
    doc_id = re.findall('\d+', tmp)
    return int(doc_id[0])

### get words/vocabularies from file
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

vocabularies = [w for i in list(x) for w in i]
# w2id = {w: ind + 2 if not(w.isdigit()) else int(w) for ind, w in enumerate(set(vocabularies))}
w2id = {}
w2id['--pad--'] = 0
w2id['--niv--'] = 1
for (i, w) in enumerate(set(vocabularies), 2):
    try:
        if(not(w.isdigit())):
            w2id[w] = i
        else:
            w2id[w] = i
    except ValueError:
        pass

# print(w2id)
id2w = {ind: w for w, ind in w2id.items()}

doc_ids = samples
doc2id = {doc_id: i for i, doc_id in enumerate(list(doc_ids))}
print(len(doc2id))

# s_train, s_test, doc_id_train, doc_id_test = [], [], [], []

id_representation = []
for doc in x:
    id_representation.append([w2id[w] for w in doc])

doc_ids_remapping = []
for doc in doc_ids:
    doc_ids_remapping.append(doc2id[doc])
# print(doc_ids_remapping)

n = 20
sentences = id_representation.copy()
for i in range(len(sentences)):
    sentences[i] = [sentences[i][j * n:(j + 1) * n] for j in range((len(sentences[i]) + n - 1) // n )] 
# print(sentences[0])

from keras.preprocessing import sequence
s = []
for i in range(len(sentences)):
    try:
        padded_s = sequence.pad_sequences(sentences[i], maxlen=n)
        s.append(np.asarray(padded_s))
        # s_and_doc_id.append(list(zip(sequence.pad_sequences(sentences[i], maxlen=n), [doc_ids[i] for j in range(len(sentences[i]))])))
    except ValueError:
        pass

flattened_sentences = np.vstack(s)
# for i in range(len(sentences)):
#     flattened_sentences += sentences[i]
#     flattened_sentences[i] = np.asarray(flattened_sentences[i])

flattened_doc_ids = []
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        flattened_doc_ids.append(doc_ids_remapping[i])

temp = list(zip(flattened_sentences, flattened_doc_ids))
# print(temp)
np.random.shuffle(temp)
flattened_doc_ids, flattened_doc_ids = list(zip(*temp))
# print(flattened_sentences)
# print(flattened_doc_ids)

flattened_doc_ids = np.array([np.asarray(s) for s in flattened_doc_ids])
# print(flattened_doc_ids)
from keras.utils import to_categorical

s_train = flattened_sentences[:int(.8 * len(flattened_sentences))]
s_test = flattened_sentences[int(.8 * len(flattened_sentences)):]
doc_id_train = flattened_doc_ids[:int(.8 * len(flattened_sentences))]
doc_id_test = flattened_doc_ids[int(.8 * len(flattened_sentences)):]

# print(to_categorical(doc_id_test, num_classes=len(doc_ids_remapping)))
# print(to_categorical(doc_id_train, num_classes=len(doc_ids_remapping)).shape)
# print(doc_id_train.shape)

### model section
from keras.layers import Activation, Bidirectional, Dense, Flatten, Embedding, InputLayer, LSTM, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model
model.add(Embedding(len(w2id), 64, input_length=n))
model.add(Bidirectional(LSTM(128)))
# model.add(Flatten())
model.add(Dense(len(doc2id), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(.0001), metrics=['accuracy'])
outputs = [layer.output for layer in model.layers]
print(outputs)
model.summary()

print(s_train[0])
model.fit(s_train, to_categorical(doc_id_train, len(doc2id)), batch_size=32, epochs=8)
scores = model.evaluate(s_test, to_categorical(doc_id_test, num_classes=len(doc2id)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")


