import deepcut
import json
import numpy as np
import os
import re

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

i = 1
for f in samples:
    dataset_path = os.path.join(path, str(f) + '.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split(',')
    doc_id = get_doc_id(data)
    print(str(i) + ' -', doc_id)
    i += 1

### group words to 20-words sentences
for sample in samples:
    [sample[i * 20:(i + 1) * 20] for i in range((len(sample) + 20 - 1) // 20 )]  
