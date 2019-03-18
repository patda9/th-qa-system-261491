from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Multiply, Subtract,Dropout, GRU,Masking, Concatenate
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sentence_length = 40
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

candidate_sentence_wv_seq = Input(shape=(sentence_length,word_vector_length,))
question_wv_seq = Input(shape=(sentence_length,word_vector_length,))

sv = sentenceVector()
print(sv.summary())
# candidate_sentence_sv = sv(candidate_sentence_wv_seq)
question_sv = sv(question_wv_seq)
rnn_sv = Input(shape=(rnn_size, ))

sc = sentenceCompare()
print(sc.summary())
similarity = sc([rnn_sv, question_sv])

# model = Model(inputs=[candidate_sentence_wv_seq, question_wv_seq], outputs=similarity)
model = Model(inputs=[rnn_sv, question_wv_seq], outputs=similarity)
print(model.summary())

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

model.load_weights('./compare2text/compare_model_v5.h5')

if __name__ == "__main__":
    # x1_train = np.load('x1_train.npy')
    x1_train = np.random.random(size=(100, rnn_size))
    # x2_train = np.load('x2_train.npy')
    x2_train = np.random.random(size=(100, sentence_length, word_vector_length))
    # y_train = np.load('y2_train.npy')
    y_train = np.random.randint(2, size=(100, ))

    print(x1_train.shape, x2_train.shape, y_train.shape)

    num_train_samples = x1_train.shape[0]

    temp = np.zeros(x1_train.shape)
    temp[:] = 0.
    # x1_train = np.concatenate((temp,x1_train),axis=1)
    x1_train = x1_train

    np.random.seed(0)
    idx = np.random.permutation(num_train_samples)
    x1_train = x1_train[idx]
    x2_train = x2_train[idx]
    y_train = y_train[idx]

    s = 0
    #y_pred = model.predict([x1_train[s:num_train_samples],x2_train[s:num_train_samples]])
    y_pred = model.predict([x1_train[s:5],x2_train[s:5]])

    i = s
    for y in y_pred:
        print(y_train[i],y)
        i = i+1

    sv_model = sentenceVector()
    sv_model.load_weights('./compare2text/compare_model_v5.h5', by_name=True)
    # x1_sv = sv.predict(x1_train[0:5,:,:])
    x1_sv = sv.predict(x1_train[0:5, :])
    x2_sv = sv.predict(x2_train[0:5,:,:])
    print(x1_sv)
    print(x2_sv)

    sc_model = sentenceCompare()
    sc_model.load_weights('./compare2text/compare_model_v5.h5', by_name=True)
    y_test = sc.predict([x1_sv,x2_sv])
    print(y_test)
