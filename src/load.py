import json
import numpy as np
import preprocessing as prep
import sentence_vectorization as sv
import os

np.random.seed(0)

path = 'C:\\Users\\Patdanai\\Desktop\\wiki-dictionary-[1-50000]\\'
text_path = 'C:\\Users\\Patdanai\\Desktop\\documents-nsc\\'
dataset = os.listdir(path)
n_samples = 512

with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
    questions = json.load(f)

with open('./../new_sample_questions_answer.json', 'r', encoding='utf-8', errors='ignore') as f:
    answer_doc_id = json.load(f)

with open('./../new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
    answer_details = json.load(f)

last_doc_id = 282972
sample_question_ans = []
selected_questions = []
for i in range(n_samples):
    randomed_question = np.random.randint(questions.__len__())
    while(answer_doc_id[randomed_question] > last_doc_id or randomed_question in selected_questions): # limited preprocessed corpus
        randomed_question = np.random.randint(questions.__len__())
    sample_question_ans.append((randomed_question + 1, answer_doc_id[randomed_question])) # question_ids start from 0 (+1)
    selected_questions.append(randomed_question)
sample_question_ans = sorted(sample_question_ans)

answer_char_locs = []
print(sample_question_ans)

for i in range(sample_question_ans.__len__()):
    for q in answer_details['data']:
        if(sample_question_ans[i][0] == q['question_id']):
            # print(sample_question_ans[i][0], q['question_id'])
            answer_begin_pos = q['answer_begin_position '] - 1 # pos - 1 to refer array index
            answer_end_pos = q['answer_end_position']
            answer_char_locs.append(range(answer_begin_pos, answer_end_pos))

article_samples = []
tokenized_article_samples = []
for t in sample_question_ans:
    text_dataset_path = os.path.join(path, str(t[1]) + '.json')
    with open(text_dataset_path, 'r', encoding='utf-8') as f:
        article = json.load(f)
    tokenized_article_samples.append(article)
    article_samples.append(''.join(article))

answers = []
for i in range(n_samples):
    answer = article_samples[i][answer_char_locs[i][0]:answer_char_locs[i][-1]]
    print('question_id:', sample_question_ans[i][0], 'answer:', answer)
    answers.append(answer)

exit()

# preprocess ของวินเนอร์ไม่เปลี่ยนตำแหน่งตัวอักษร
# หาคำที่เป็นคำตอบก่อนจะได้ ['รา'] -> [468:469], ['บัต '] -> [470:473] แล้วค่อย remove stop words
# {
#   "question_id":3994,
#   "question":"ปัตตานี เป็นจังหวัดในภาคใดของประเทศไทย",
#   "answer":"ใต้","answer_begin_position ":125,
#   "answer_end_position":128,
#   "article_id":6865
# }

# q = [w for w in questions if w.strip()]

## load input question sentence and output answer sentence
for i in range(questions.__len__()):
    questions[i] = [w for w in questions[i] if not(w is ' ')]
    # print(i, questions[i], answer_doc_id[i])

# get article ids from directory path


## get words/vocabularies from file
articles = []
i = 1
for t in sample_question_ans:
    dataset_path = os.path.join(path, str(t[1]) + '.json')

    with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
        data = prep.remove_xml(data)
        data = prep.remove_stop_words(data)
    articles.append(data)
    i += 1

# vocabularies = [word for article in list(articles) for word in article]
# vocabs_out = {}

# for i in range(vocabularies.__len__()):
#     vocabs_out[i] = vocabularies[i]

# with open('./vocabs.json', 'w', encoding='utf-8', errors='ignore') as o:
#     json.dump(vocabs_out, o)

with open('./vocabs.json', 'r', encoding='utf-8', errors='ignore') as f:
    vocabs = json.load(f).values()

# print(vocabs)

w2id = {}
w2id['--niv--'] = 0
for (i, w) in enumerate(set(vocabs), 1):
    try:
        if(not(w.isdigit())):
            w2id[w] = i
        else:
            w2id[w] = i
    except ValueError:
        w2id[w] = 0

id2w = {idx: w for w, idx in w2id.items()}

print(id2w[0])

id_representation = []
for article in articles:
    try:
        id_representation.append([w2id[w] for w in article])
    except KeyError:
        id_representation.append([w2id['--niv--'] for w in article])

print([id2w[idx] for idx in id_representation[499]])

words_per_sentence = 20
overlap = False

sv.k_words_separate(words_per_sentence, id_representation, overlap=overlap)

from keras.layers import Activation, Bidirectional, Dense, Flatten, Embedding, InputLayer, LSTM, Masking, TimeDistributed
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam

