# utils
import json
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix

# numerical libs
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Bidirectional, Concatenate, Dense, Flatten, GRU, Input, Lambda, LSTM, Masking, multiply, Permute, RepeatVector
from keras.models import load_model, Model, Sequential

hidden_nodes = 16
rnn_units = 64
sl = 40
wvl = 300

def attention_layer(inputs, time_step):
    a = Lambda(lambda x: x, output_shape=lambda s: s)(inputs)
    a = Permute((2, 1))(a)
    a = Dense(time_step, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention = multiply([inputs, a_probs], name='multiply_attention')
    return output_attention

def get_input(input_path):
    return np.load(input_path)

def sequence_generator(files, label_path, batch_size=512):
    # files => os.listdir
    if(label_path):
        positive_dataset_path = label_path[0]
        negative0_dataset_path = label_path[1]
        negative1_dataset_path = label_path[2]
    
    while(1):
        batch_paths = np.random.choice(files, size=batch_size)
        input_batch = np.empty((0, 20, 300))
        output_batch = np.empty((0, 1))

        inp = None
        for input_path in batch_paths:
            if('positive' in input_path):
                inp = get_input(positive_dataset_path + input_path)
                out = np.ones((inp.shape[0], 1))
            elif('negative0' in input_path):
                inp = get_input(negative0_dataset_path + input_path)
                out = np.zeros((inp.shape[0], 1))
            elif('negative1' in input_path):
                inp = get_input(negative1_dataset_path + input_path)
                out = np.zeros((inp.shape[0], 1))

            input_batch = np.concatenate((input_batch, inp), axis=0)
            output_batch = np.concatenate((output_batch, out), axis=0)

        input_batch = np.array(input_batch)
        output_batch = np.array(output_batch)

        yield (input_batch, output_batch)

def sentence_vector(recurrent_layer):
    input_seq = Input(shape=(sl, wvl))
    masking = Masking(mask_value=0., input_shape=(sl, wvl))(input_seq)
    rl = recurrent_layer(masking)
    multiply_attention = attention_layer(rl, sl)
    output = Dense(rnn_units, activation='relu')(multiply_attention)
    submodel = Model(inputs=input_seq, outputs=output)
    submodel.summary()
    return submodel

def sentence_compare():
    qv = Input(shape=(rnn_units, ), name='vectorized_q')
    sv = Input(shape=(rnn_units, ), name='vectorized_s')

    concatenate = Concatenate()([qv, sv])
    dense1 = Dense(hidden_nodes, activation='sigmoid')(concatenate)
    dense2 = Dense(hidden_nodes, activation='sigmoid')(dense1)
    similarity = Dense(1, activation='sigmoid')(dense2)
    submodel = Model(inputs=[qv, sv], outputs=similarity)
    submodel.summary()
    return submodel

if __name__ == "__main__":
    # create tensors 
    q_seq = Input(shape=(sl, wvl), name='question_seq')
    s_seq = Input(shape=(sl, wvl), name='sentence_seq')

    # create recurrrent layers
    rl1 = Bidirectional(LSTM(rnn_units, activation='relu', dropout=.4, recurrent_dropout=.1, return_sequences=1))
    rl2 = Bidirectional(LSTM(rnn_units, activation='relu', dropout=.4, recurrent_dropout=.1, return_sequences=1))

    # create output tensors
    qv = sentence_vector(rl1)
    qv.name = 'question_vector'
    qv = qv(q_seq)

    sv = sentence_vector(rl2)
    sv.name = 'sentence_vector'
    sv = sv(s_seq)

    similarity = sentence_compare()
    similarity.name = 'compare'
    similarity = similarity([qv, sv])

    # form model
    model = Model(inputs=[q_seq, s_seq], outputs=similarity)
    model.summary()
