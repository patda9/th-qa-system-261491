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
    return [data[i * n:(i + 1) * n] for i in range((len(data) + n - 1) // n )] 

i = 1
for f in samples:
    dataset_path = os.path.join(path, str(f) + '.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(data)
        for j in range(len(data)):
            data[j] = data[j].replace('"', '').strip(' ')
            if(data[j] == ''):
                data[j] = data[j].replace('', ' ') # remove double quotes from data
    # print('*****', i, (data))

    doc_id = get_doc_id(data)
    data = n_word_sentence_segment(20, data)
    # print(str(i) + ' -', doc_id)
    i += 1

data = np.asarray(data)
print(data)
samples = np.asarray(samples).reshape((len(samples), 1))
y = doc_ids = samples
print(y)

# json.dump(data, open('test.txt', 'w', encoding='utf-8-sig'), ensure_ascii=True) # save sentence
for i in range(len(y)):
    for j in range(len(data)):
        print(i, [data[j], y[i]])
