import json
import matplotlib.pyplot as plt
import numpy as np

from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, \
GRU, Input, LSTM, Masking, Multiply, Subtract
from keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix

rnn_size = 16
sentence_length = 40
word_vector_length = 100

x1_train = np.load('C:/Users/Patdanai/Downloads/train/x1_train.npy')
x2_train = np.load('C:/Users/Patdanai/Downloads/train/x2_train.npy')
y_train = np.load('C:/Users/Patdanai/Downloads/train/y2_train.npy')
print(x1_train.shape, x2_train.shape, y_train.shape, sep='\n')

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

# create tensors
candidate_sentence_wv_seq = Input(shape=(sentence_length, word_vector_length,))
question_wv_seq = Input(shape=(sentence_length,word_vector_length,))

# form model
sv = sentenceVector()
candidate_sentence_sv = sv(candidate_sentence_wv_seq)
question_sv = sv(question_wv_seq)
dissimilarity = sentenceCompare()([candidate_sentence_sv, question_sv])
model = Model(inputs=[candidate_sentence_wv_seq, question_wv_seq], outputs=dissimilarity)
print(model.summary())
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['mae','accuracy'])

# pad with np.zeros => (?, 40, 100)
num_train_samples = x1_train.shape[0]
temp = np.zeros(x1_train.shape)
temp[:] = 0.
x1_train = np.concatenate((temp, x1_train), axis=1)
temp = np.zeros((x2_train.shape[0], sentence_length - x2_train.shape[1], word_vector_length))
temp[:] = 0.
x2_train = np.concatenate((temp, x2_train), axis=1)
print('y:', y_train.shape, 'x1:', np.array(x1_train).shape, 'x2:', np.array(x2_train).shape)

# shuffle x1, x2, y with same order
np.random.seed(5)
idx = np.random.permutation(num_train_samples)
x1_train = x1_train[idx]
x2_train = x2_train[idx]
y_train = y_train[idx]

# create class weight
class_weight = {0: 1.0,
                1: 4.0}

# create callback
file_path = './models/compare_model_v3/{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callbacks_list = [checkpoint]

history = model.fit([x1_train, x2_train], y_train, batch_size=2048, callbacks=callbacks_list, class_weight=class_weight, epochs=256, validation_split=0.2)

model.save('compare_model_v3.h5')
# TODO save sentence vectors from SV_BLOCK

s = .8 * num_train_samples
y_pred = model.predict([x1_train[s:num_train_samples],x2_train[s:num_train_samples]])

i = s
for y in y_pred:
    print(y_train[i],y)
    i = i+1

cm = confusion_matrix(y_train[s:num_train_samples] >= 0.5, y_pred >= 0.5)
np.savetxt('./results/new_model_cm.txt', cm, fmt='%d')

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.savefig('./results/new_model_loss.png')
plt.show()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.savefig('./results/new_model_acc.png')
plt.show()

# sv_model = sentenceVector()
# sv_model.load_weights('compare_model_v3.h5', by_name=True)
# print(sv_model.summary())

# sc_model = sentenceCompare()
# sc_model.load_weights('compare_model_v3.h5', by_name=True)
# print(sc_model.summary())
