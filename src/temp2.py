from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Multiply, Subtract,Dropout, GRU, Masking, Concatenate, BatchNormalization
from preprocessing import m_words_separate
import numpy as np
import os

sentence_length = 40
word_per_sentence = 20
word_vector_length = 100
rnn_size = 32

def sentenceVector():
    submodel = Sequential()
    submodel.add(Masking(mask_value=0., input_shape=(sentence_length, word_vector_length, )))
    submodel.add(GRU(rnn_size,activation='relu',name='sv_rnn1'))
    submodel.add(Dropout(0.5))
    return submodel

def sentenceCompare():
    candidate_sentence_sv = Input(shape=(rnn_size,))
    question_sv = Input(shape=(rnn_size,))
    concate = Concatenate()([candidate_sentence_sv, question_sv])
    dense1 = Dense(32, activation='sigmoid',name='sc_dense1')(concate)
    dense2 = Dense(32, activation='sigmoid',name='sc_dense2')(dense1)
    similarity = Dense(1, activation='sigmoid',name='sc_dense3')(dense2)
    submodel = Model(inputs=[candidate_sentence_sv, question_sv], outputs=similarity)
    return submodel

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fscore(y_true, y_pred):
    beta = 1
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

candidate_sentence_wv_seq = Input(shape=(sentence_length,word_vector_length,))
question_wv_seq = Input(shape=(sentence_length,word_vector_length,))

sv_model = sentenceVector()
sv_model.load_weights('./compare2text/compare_model_v5.h5', by_name=True)

PATH = 'D:/Users/Patdanai/th-qasys-db/corpus_wv/'
OUT_PATH = 'D:/Users/Patdanai/th-qasys-db/corpus_sv/'
files = os.listdir(PATH)

cont = True
data = None

# temp_counter = 50304 # for continue writing
temp_counter = 0
file_number = temp_counter + 1
for i in range(temp_counter, len(files)):
    # extract doc id
    f_name = ''.join([c for c in files[i] if c.isdigit()])
    data = np.load(PATH + files[i])
    # print(data.shape)

    # [0][0] to reduce dimension from batch processing based
    data = m_words_separate(word_per_sentence, [data], overlapping_words=word_per_sentence//2)[0][0]
    
    # pad np.zeros in front of data to make data.shape becomes (?, 40, 100)
    temp = np.zeros((data.shape[0], sentence_length - data.shape[1], word_vector_length))
    temp[:] = 0.
    data = np.concatenate((temp, data), axis=1)
    # print(data.shape)

    # vectorize sentence
    data = sv_model.predict(data)

    # save to .npy format
    np.save(OUT_PATH + 'sv-' + files[i], data)
    print('[%d/%d]\tShape:%s\tSaved\t%s\tto\t%s. \r' % (file_number, len(files), str(data.shape), 'sv-' + files[i], OUT_PATH))
    file_number += 1
data = None
