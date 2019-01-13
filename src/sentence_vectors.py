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
_, flattened_doc_ids = list(zip(*temp))
print(flattened_sentences)
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

print(doc_id_test.shape)

a = 50
for s in s_test[a:100]:
    id_list = []
    s_list = []
    for i in s:
        id_list.append(i)
        s_list.append(id2w[i])
    print(id_list, doc_id_test[a])
    print(s_list, doc_id_test[a])
    a += 1

### model section
from keras.layers import Activation, Bidirectional, Dense, Flatten, Embedding, InputLayer, LSTM, TimeDistributed
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(len(w2id), 64, input_length=n))
model.add(Bidirectional(LSTM(128)))
# model.add(Flatten()) # for return_sequence=True
model.add(Dense(32, activation='relu'))
model.add(Dense(len(doc2id), activation='softmax'))

# model = load_model('./src/sentence-vector-model.h5')
model.compile(loss='categorical_crossentropy', optimizer=Adam(.0001), metrics=['accuracy'])
model.summary()

# model.fit(s_train, to_categorical(doc_id_train, len(doc2id)), batch_size=32, epochs=8)
# model.save('./sentence-vector-model.h5')
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(index=2).output)
# json_architecture = model.to_json()

scores = model.evaluate(s_test, to_categorical(doc_id_test, num_classes=len(doc2id)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

predict_sample = np.asarray([s_test])
dense1_output = intermediate_layer_model.predict(np.asarray(s_test))

def generate_question(words, n=20, padding=False):
    question_in_id = []
    for w in words:
        try:
            question_in_id.append(w2id[w])
        except KeyError:
            question_in_id.append(w2id['--niv--'])
    if(padding == True):
        while(len(question_in_id) < 20):
            question_in_id.insert(0, w2id['--pad--'])
    return question_in_id

question_sentence = generate_question(['ใน', 'แต่ละ', 'ทีม', 'จะ', 'ต้อง', 'ประกอบ', 'อะไร', 'เพื่อ', 'นำ', 'ไป', 'ติด', 'ที่', 'แท่น'], padding=True)
question_vector = intermediate_layer_model.predict(np.asarray([question_sentence]))


print(question_sentence)

answer_sentence = s_test[126]
answer_vector = intermediate_layer_model.predict(np.asarray([answer_sentence]))

print(dense1_output.shape)

shortest_sentences = []
shortest_idx = []

for i in range(len(s_test)):
    dist = np.linalg.norm(question_vector[0] - dense1_output[i])

    if(len(shortest_sentences) < 50):
        shortest_sentences.append(dist)
        shortest_idx.append(i)

    else:
        if(np.amax(shortest_sentences) > dist):
            idx = np.argmax(shortest_sentences)
            shortest_sentences[idx] = dist
            shortest_idx[idx] = i

for i in range(len(shortest_sentences)):
    s = s_test[shortest_idx[i]]
    print(s)
    w_list = []
    for w in s:
        w_list.append(id2w[w])
    print(w_list, doc_id_test[shortest_idx[i]])


dist = np.linalg.norm(question_vector[0] - answer_vector[0])
print(shortest_sentences)
print(shortest_idx)
print(dist)

# print(question_sentence)
# print(question_vector)
# similar_sentence = np.array([w2id['โดย'], w2id[' '], w2id['5'], w2id[' '], w2id['ตาม'], w2id['ลำดับ'], w2id[' '], w2id['แต่'], w2id['ทั้ง'], w2id['สอง'], w2id['ทีม'], w2id['ใช้'], w2id['เส้นทาง'], w2id['พิเศษ'], w2id['ที่'], w2id['ผิด'], w2id['กฎจราจร'], w2id[' '], w2id['โดย'], w2id['ขับ'], ])
# dense_output2 = intermediate_layer_model.predict(np.asarray([s_test[1111]]))
# similar_output = intermediate_layer_model.predict(np.asarray([similar_sentence]))
# print(similar_sentence)
# file = open('diff_vec.txt', 'w')
# file.writelines(str(dense_output.tolist()))
# file.writelines(str(similar_output.tolist()))
# file.writelines(str(dense_output2.tolist()))
# file.close()
# file = open('sentences.txt', 'w', 'utf-8')
# file.close()

prediction = model.predict(np.asarray(s_test))
file = open('sentence-vector-outputs1.txt', 'w')
file.writelines(str(dense1_output.tolist()))
file.close()

exit()
# print(len(prediction))
# for i in range(len(prediction)):
#     if(i % 25 == 0):
#         print(predict_sample[i], 'predict:', np.argmax(prediction[i]), 'g-truth:', doc_id_test[i])
#         # for j in range(len(predict_sample[i])):
#         #     print(id2w[predict_sample[i][j]])
