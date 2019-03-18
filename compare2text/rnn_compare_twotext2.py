from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Multiply, Subtract,Dropout, GRU,Masking,Concatenate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras.backend as K

from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

sentence_length = 40
word_vector_length = 100
rnn_size = 32


def sentenceVector():
    submodel = Sequential()
    submodel.add(Input(shape=(sentence_length, word_vector_length)))
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
plot_model(sv, to_file='sv.png')
candidate_sentence_sv = sv(candidate_sentence_wv_seq)
question_sv = sv(question_wv_seq)

sc = sentenceCompare()
similarity = sc([candidate_sentence_sv, question_sv])

model = Model(inputs=[candidate_sentence_wv_seq, question_wv_seq], outputs=similarity)

print(model.summary())

plot_model(sv, to_file='sv.png')
plot_model(sc, to_file='sc.png')
plot_model(model, to_file='model.png')
exit()

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


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['mae','accuracy',precision,recall,fscore])

x1_train = np.load('x1_train.npy')
x2_train = np.load('x2_train.npy')
y_train = np.load('y2_train.npy')

num_train_samples = x1_train.shape[0]

temp = np.zeros(x1_train.shape)
temp[:] = 0.
x1_train = np.concatenate((temp,x1_train),axis=1)

np.random.seed(0)
idx = np.random.permutation(num_train_samples)
x1_train = x1_train[idx]
x2_train = x2_train[idx]
y_train = y_train[idx]

class_weight = {0: 1.0,
                1: 5.0}

history = model.fit([x1_train,x2_train],y_train,epochs=100,batch_size=100,validation_split=0.2, class_weight=class_weight)

model.save('compare_model_v6.h5')

s = int(0.8*num_train_samples)
y_pred = model.predict([x1_train[s:num_train_samples],x2_train[s:num_train_samples]])

#i = s
#for y in y_pred:
#    print(y_train[i],y)
#    i = i+1

cm = confusion_matrix(y_train[s:num_train_samples] >= 0.5, y_pred >= 0.5)
print(cm)

fig, axes = plt.subplots(nrows=2, ncols=3)
axes[0, 0].plot(history.history['loss'],'b')
axes[0, 0].plot(history.history['val_loss'],'r')
axes[0, 0].set_title("Loss")

axes[0, 1].plot(history.history['mean_absolute_error'],'b')
axes[0, 1].plot(history.history['val_mean_absolute_error'],'r')
axes[0, 1].set_title("MAE")

axes[0, 2].plot(history.history['acc'],'b')
axes[0, 2].plot(history.history['val_acc'],'r')
axes[0, 2].set_title("Accuracy")

axes[1, 0].plot(history.history['precision'],'b')
axes[1, 0].plot(history.history['val_precision'],'r')
axes[1, 0].set_title("Precision")

axes[1, 1].plot(history.history['recall'],'b')
axes[1, 1].plot(history.history['val_recall'],'r')
axes[1, 1].set_title("Recall")

axes[1, 2].plot(history.history['fscore'],'b')
axes[1, 2].plot(history.history['val_fscore'],'r')
axes[1, 2].set_title("F-score")

plt.show()

#sv_model = sentenceVector()
#sv_model.load_weights('compare_model_v3.h5', by_name=True)
#print(sv_model.summary())

#sc_model = sentenceCompare()
#sc_model.load_weights('compare_model_v3.h5', by_name=True)
#print(sc_model.summary())

#[[4929 1556]
# [ 288  993]]





