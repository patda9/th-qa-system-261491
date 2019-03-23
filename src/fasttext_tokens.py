# utility libs
import json
import os

# numerical libs
import numpy as np

PATH_TO_TOKENS = 'F:/tokens_nsc/'

# for f_name in os.listdir(PATH_TO_TOKENS):
#     print(f_name)
#     exit()
with open('../665.json', encoding='utf-8') as f:
    data = json.load(f)
    title = data[0]
    tokens = data[1]

temp = []
i = 0
while(i < len(tokens)):
    try:
        if(tokens[i] == ' ' or tokens[i] == ';'):
            pass
        elif(tokens[i].isdigit and tokens[i+1] == '.' and tokens[i+2].isdigit):
            temp.append(tokens[i] + tokens[i+1] + tokens[i+2])
            i += 2
        else:
            temp.append(tokens[i])
    except:
        pass
    i += 1

vocab = set([w for w in temp])
# print(vocab)

vocab_vectors = {}
fasttext_vec_file = open('C:/Users/Patdanai/Desktop/261499-nlp/lab/cc.th.300.vec', 'r', encoding='utf-8-sig')

wvl = 300
count = 0
for line in fasttext_vec_file:
    if count > 0:
        line = line.split()
        if(line[0] in vocab):
            vocab_vectors[line[0]] = line[1:]
            break
    count = count + 1

word_vectors = np.zeros((len(temp), wvl))
print(word_vectors.shape)

for i in range(len(temp)):
    try:
        print(temp[i])
        word_vectors[i, :] = vocab_vectors[temp[i]]
    except:
        word_vectors[i, :] = word_vectors[i]
    print(word_vectors[i])

# print(temp)