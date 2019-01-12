import json
import numpy as np
import preprocessing as prep

# TODO
# load question samples
# load answer samples
# load answer docs
# load answer positions

# get article ids from directory path
# _____

## load input qusetion and output doc
with open('./new_sample_questions_tokenize.json', 'r', encoding='utf-8') as f1:
    questions = json.load(f1)

with open('./new_sample_questions_answer.json', 'r', encoding='utf-8') as f2:
    answer_doc_id = json.load(f2)

for i in range(len(questions)):
    print(str(i) + ':', questions[i], answer_doc_id[i])

n = 20
sentences = []

for i in range(len(sentences)):
    sentences[i] = [sentences[i][j * n:(j + 1) * n//2] for j in range((len(sentences[i]) + n - 1) // n )]
print(sentences)

# from keras.models import Sequential

model = None