from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Multiply, Subtract,Dropout, GRU, Masking, Concatenate, BatchNormalization
import numpy as np

rnn_size = 32
sentence_length = 40
word_vector_length = 100

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

gru_sv = Input(shape=(rnn_size, )) # create tensor: sentence vector(s) database

qsv_model = sentenceVector() # submodel
qsv_model.load_weights('./compare2text/compare_model_v5.h5', by_name=True) # load submodel weight
qs_seq = Input(shape=(sentence_length, word_vector_length)) # create tensor
qsv_model = qsv_model(qs_seq) # add input tensor: question sequence
similarity = sentenceCompare()([gru_sv, qsv_model]) # add input tensors: candidate sv, question sv

# form model
model = Model(inputs=[gru_sv, qs_seq], outputs=similarity)
print(model.summary())

exit()

#

rnn_size = 16
sentence_length = 40
word_vector_length = 100

def sentenceVector(): # SV_BLOCK
    submodel = Sequential()
    submodel.add(Masking(mask_value=0., input_shape=(sentence_length, word_vector_length, )))
    submodel.add(GRU(rnn_size, activation='relu', name='sv_rnn1'))
    submodel.add(BatchNormalization())
    submodel.add(Dropout(0.4))
    return submodel

def sentenceCompare(): # SC_BLOCK
    candidate_sentence_sv = Input(shape=(rnn_size,))
    question_sv = Input(shape=(rnn_size,))
    concate = Concatenate()([candidate_sentence_sv, question_sv])
    dense1 = Dense(16, activation='sigmoid',name='sc_dense1')(concate)
    batch_norm_sequences = BatchNormalization()(dense1)
    dense2 = Dense(16, activation='sigmoid',name='sc_dense2')(dense1)
    batch_norm_sequences = BatchNormalization()(dense2)
    dissimilarity = Dense(1, activation='sigmoid',name='sc_dense3')(dense2)
    submodel = Model(inputs=[candidate_sentence_sv, question_sv], outputs=dissimilarity)
    return submodel

qsv_model = sentenceVector()
qsv_model.load_weights('compare_model_v3.h5', by_name=True)
print(qsv_model.summary())

question_wv_seq = Input(shape=(sentence_length,word_vector_length,))
vectorized_qs = qsv_model(question_wv_seq)
candidate_sentence_seq = Input(shape=(rnn_size,))

sc_model = sentenceCompare()
sc_model.load_weights('compare_model_v3.h5', by_name=True)
print(sc_model.summary())

similarity = sc_model([candidate_sentence_seq, vectorized_qs])

model = Model(inputs=[candidate_sentence_seq, question_wv_seq], outputs=similarity)
print(model.summary())

# c = np.ones((4, rnn_size))
# q = np.ones((4, sentence_length, word_vector_length))

prediction = model.predict([c, q])

print(prediction)
