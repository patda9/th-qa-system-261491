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

# seed
np.random.seed(0)

# file i/o paths
p_path = 'D:/Users/Patdanai/th-qasys-db/positive_embedded/positive_embedded/'
n0_path = 'D:/Users/Patdanai/th-qasys-db/n0_embedded/n0_embedded/'
n1_path = 'D:/Users/Patdanai/th-qasys-db/n1_embedded/n1_embedded/'
dataset_paths = [p_path, n0_path, n1_path]

# model hyperparameters
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

def fscore(y_true, y_pred):
    beta = 1
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def get_input(input_path):
    return np.load(input_path)

def get_training_set(dataset_paths, batch_size=4000, samples_per_file=5):
    if(samples_per_file > 5):
        print('samples per file must be in range [0, 5]')
        return

    files = np.random.permutation(os.listdir(dataset_paths[0]))
    
    for f_name in files:
        q_idx = f_name.replace('positive_question', '').replace('.npy', '')
        print(q_idx)
        
        positive_s = get_input(dataset_paths[0] + f_name)[:samples_per_file]
        print(positive_s.shape)
        exit()

    return 1

def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues,
                        normalize=0, save_path=None, title=None, verbose=1):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting ``normalize=True``.
    """

    if(not title):
        if(normalize):
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    if(normalize):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if(verbose):
        print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
            rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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

    training_set = get_training_set(dataset_paths, samples_per_file=1)
