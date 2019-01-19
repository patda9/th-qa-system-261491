import json
import numpy as np
import preprocessing as prep
import os
np.random.seed(0)

path = 'C:\\Users\\Patdanai\\Desktop\\wiki-dictionary-[1-50000]\\'
text_path = 'C:\\Users\\Patdanai\\Desktop\\documents-nsc\\'
dataset = os.listdir(path)
n_samples = 500

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

from keras.layers import Activation, Bidirectional, Dense, Flatten, Embedding, InputLayer, LSTM, Masking, TimeDistributed
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam

model = Sequential()
# model.add(Masking(mask_value=0., ()))
model.add(Embedding(len(w2id), 64, input_length=n))
model.add(Bidirectional(LSTM(128)))
# # model.add(Flatten()) # for return_sequence=True
# model.add(Dense(32, activation='relu'))
# model.add(Dense(len(doc2id), activation='softmax'))

# # model = load_model('./src/sentence-vector-model.h5')
# model.compile(loss='categorical_crossentropy', optimizer=Adam(.0001), metrics=['accuracy'])
# model.summary()

# # model.fit(s_train, to_categorical(doc_id_train, len(doc2id)), batch_size=32, epochs=8)
# # model.save('./sentence-vector-model.h5')
# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(index=2).output)
# # json_architecture = model.to_json()

# scores = model.evaluate(s_test, to_categorical(doc_id_test, num_classes=len(doc2id)))
# print(f"{model.metrics_names[1]}: {scores[1] * 100}")

# predict_sample = np.asarray([s_test])
# dense1_output = intermediate_layer_model.predict(np.asarray(s_test))

# def generate_question(words, n=20, padding=False):
#     question_in_id = []
#     for w in words:
#         try:
#             question_in_id.append(w2id[w])
#         except KeyError:
#             question_in_id.append(w2id['--niv--'])
#     if(padding == True):
#         while(len(question_in_id) < 20):
#             question_in_id.insert(0, w2id['--pad--'])
#     return question_in_id

# question_sentence = generate_question(['ใน', 'แต่ละ', 'ทีม', 'จะ', 'ต้อง', 'ประกอบ', 'อะไร', 'เพื่อ', 'นำ', 'ไป', 'ติด', 'ที่', 'แท่น'], padding=True)
# question_vector = intermediate_layer_model.predict(np.asarray([question_sentence]))


# print(question_sentence)

# answer_sentence = s_test[126]
# answer_vector = intermediate_layer_model.predict(np.asarray([answer_sentence]))

# print(dense1_output.shape)

# shortest_sentences = []
# shortest_idx = []

# for i in range(len(s_test)):
#     dist = np.linalg.norm(question_vector[0] - dense1_output[i])

#     if(len(shortest_sentences) < 50):
#         shortest_sentences.append(dist)
#         shortest_idx.append(i)

#     else:
#         if(np.amax(shortest_sentences) > dist):
#             idx = np.argmax(shortest_sentences)
#             shortest_sentences[idx] = dist
#             shortest_idx[idx] = i

# for i in range(len(shortest_sentences)):
#     s = s_test[shortest_idx[i]]
#     print(s)
#     w_list = []
#     for w in s:
#         w_list.append(id2w[w])
#     print(w_list, doc_id_test[shortest_idx[i]])


# dist = np.linalg.norm(question_vector[0] - answer_vector[0])
# print(shortest_sentences)
# print(shortest_idx)
# print(dist)

# # print(question_sentence)
# # print(question_vector)
# # similar_sentence = np.array([w2id['โดย'], w2id[' '], w2id['5'], w2id[' '], w2id['ตาม'], w2id['ลำดับ'], w2id[' '], w2id['แต่'], w2id['ทั้ง'], w2id['สอง'], w2id['ทีม'], w2id['ใช้'], w2id['เส้นทาง'], w2id['พิเศษ'], w2id['ที่'], w2id['ผิด'], w2id['กฎจราจร'], w2id[' '], w2id['โดย'], w2id['ขับ'], ])
# # dense_output2 = intermediate_layer_model.predict(np.asarray([s_test[1111]]))
# # similar_output = intermediate_layer_model.predict(np.asarray([similar_sentence]))
# # print(similar_sentence)
# # file = open('diff_vec.txt', 'w')
# # file.writelines(str(dense_output.tolist()))
# # file.writelines(str(similar_output.tolist()))
# # file.writelines(str(dense_output2.tolist()))
# # file.close()
# # file = open('sentences.txt', 'w', 'utf-8')
# # file.close()

# prediction = model.predict(np.asarray(s_test))
