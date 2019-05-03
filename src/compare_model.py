# utils
import datetime
import json
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix

# numerical libs
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Bidirectional, Concatenate, Dense, Flatten, GRU, Input, Lambda, LSTM, Masking, multiply, Permute, RepeatVector
from keras.models import load_model, Model, Sequential

# seed and precision
np.random.seed(0)
np.set_printoptions(precision=4)

# file i/o paths
q_path = './data/dataset/questions/embedded_questions_4000_40_300.npy'
n0_path = 'D:/Users/Patdanai/th-qasys-db/n0_embedded/n0_embedded/'
n1_path = 'D:/Users/Patdanai/th-qasys-db/n1_embedded/n1_embedded/'
p_path = 'D:/Users/Patdanai/th-qasys-db/positive_embedded/positive_embedded/'
dataset_paths = [p_path, n0_path, n1_path, q_path]

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

def get_dataset_set(dataset_paths, batch_size=4000, samples_per_file=5):
    if(samples_per_file > 5 or not samples_per_file):
        print('samples per file must be in range [0, 5]')
        return

    question_sentences = get_input(dataset_paths[3])
    print(question_sentences.shape)

    files = np.random.permutation(os.listdir(dataset_paths[0]))
    
    s_seq = (0, sl, wvl)
    l_shape = (0, 1)

    questions_batch = np.empty((s_seq))
    sentences_batch = np.empty((s_seq))
    labels_batch = np.empty((l_shape))

    f_count = 0
    for f_name in files:
        if(f_count > batch_size - 1):
            break

        q_idx = f_name.replace('positive_question', '').replace('.npy', '')

        q = question_sentences[int(q_idx)]
        q = np.expand_dims(q, axis=0)

        positive_s = get_input(dataset_paths[0] + f_name)[:samples_per_file]
        l = np.ones((positive_s.shape[0], 1))
        labels_batch = np.concatenate((labels_batch, l), axis=0)

        negative_s = np.empty((s_seq))
        if(samples_per_file < 2):
            r = np.random.randint(2)
            if(r):
                negative0_s = get_input(dataset_paths[1] + 'negative0_question%s.npy' % q_idx)[:1]
                negative_s = np.concatenate((negative_s, negative0_s), axis=0)
            else:
                negative1_s = get_input(dataset_paths[2] + 'negative1_question%s.npy' % q_idx)[:1]
                negative_s = np.concatenate((negative_s, negative1_s), axis=0)
        else:
            n0_instance = samples_per_file // 2
            negative0_s = get_input(dataset_paths[1] + 'negative0_question%s.npy' % q_idx)[:n0_instance]
            n1_instance = samples_per_file - negative0_s.shape[0]
            negative1_s = get_input(dataset_paths[2] + 'negative1_question%s.npy' % q_idx)[:n1_instance]
            
            temp = np.concatenate((negative0_s, negative1_s), axis=0)
            negative_s = np.concatenate((negative_s, temp), axis=0)
        
        temp_s = np.concatenate((positive_s, negative_s), axis=0)
        sentences_batch = np.concatenate((sentences_batch, temp_s), axis=0)
        l = np.zeros((negative_s.shape[0], 1))
        labels_batch = np.concatenate((labels_batch, l), axis=0)

        temp_q = np.repeat(q, positive_s.shape[0] + negative_s.shape[0], axis=0)
        questions_batch = np.concatenate((questions_batch, temp_q), axis=0)

        f_count += 1

        print('file[%s/%s] => ps_shape = %s ns_shape = %s ps_ns_shape = %s l_shape = %s qs_shape = %s' % (f_count, batch_size, positive_s.shape, negative_s.shape, sentences_batch.shape, labels_batch.shape, questions_batch.shape))

    return ([questions_batch, sentences_batch], labels_batch)

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
    
    if(save_path):
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

    return ax

def plot_training_history(h, labels=['Training', 'Validation'], yl=None, save_path=None):
    plt.plot(h[0], label=labels[0])
    plt.plot(h[1], '-r', label=labels[1])
    plt.legend(loc='best')
    plt.xlabel('Epochs')

    if(yl):
        plt.ylabel(yl)
    
    if(save_path):
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

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
    multiply_attention = Lambda(lambda x: x, output_shape=lambda s: s)(multiply_attention)
    multiply_attention = Flatten()(multiply_attention)
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

current_model = 'lstm'
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
    
    # training configuration
    ##    model confiuration
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'acc'])

    ##    load model
    # model = load_model('./model/trained_compare_model_2019_05_03_14_55_55.705298.h5')

    ##   dataset
    batch_size = 2048
    samples_per_file = 5
    validation_proportion = .8
    
    ##    model fitting
    epochs = 64
    training_batch_size = 2048
    validation_split = .4

    # generate training set
    training_set = get_dataset_set(dataset_paths, batch_size=batch_size, samples_per_file=samples_per_file)

    training_size = int(len(training_set[1]) * validation_proportion)
    y_train = training_set[1][:training_size]
    x_train = [training_set[0][0][:training_size], training_set[0][1][:training_size]]

    y_test = training_set[1][training_size:]
    x_test = [training_set[0][0][training_size:], training_set[0][1][training_size:]]

    print('train on', len(y_train), 'samples')
    print('validate on', len(y_test), 'samples')

    

    print('model training process.')
    history = model.fit(x_train, y_train, batch_size=training_batch_size, epochs=epochs, validation_split=validation_split)

    dt = datetime.datetime.now()
    dt = str(dt).replace(':', '_').replace(' ', '_').replace('-', '_')

    with open('./training/history/%s_log_%s_%s.json' % (current_model, dt, batch_size), 'w') as fp:
        json.dump(history.history, fp)

    model.save('./model/%s_compare_model_%s_%s.h5' % (current_model, dt, batch_size))

    # test = get_dataset_set(dataset_paths, batch_size=32, samples_per_file=5)
    # y_test = test[1]
    # x_test = [test[0][0], test[0][1]]

    # model prediction
    class_names = ['Similar', 'Dissimilar']
    predictions = model.predict(x_test)

    positive_predictions = []
    negative_predictions = []

    for i in range(len(predictions)):
        if(y_test[i][0] == 1.):
            positive_predictions.append(predictions[i][0])
        else:
            negative_predictions.append(predictions[i][0])

    # plot histogram of prediction values
    n_bins = 10
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(positive_predictions, n_bins)
    axs[0].set_title('Positive')
    axs[1].hist(negative_predictions, n_bins)
    axs[1].set_title('Negative')
    plt.savefig('./training/distribution/%s_pred_dist_%s_%s.png' % (current_model, dt, batch_size))
    plt.clf()

    # plot training history
    acc = [history.history['acc'], history.history['val_acc']]
    plot_training_history(acc, save_path='./training/acc/%s_acc_%s_%s.png' % (current_model, dt, batch_size), yl='Accuracy',)

    loss = [history.history['loss'], history.history['val_loss']]
    plot_training_history(loss, save_path='./training/loss/%s_loss_%s_%s.png' % (current_model, dt, batch_size), yl='Loss (Binary Crossentropy)')

    error = [history.history['mean_absolute_error'], history.history['val_mean_absolute_error']]
    plot_training_history(error, save_path='./training/error/%s_error_%s_%s.png' % (current_model, dt, batch_size), yl='MAE')

    # confusion matrix of actual and predict
    plot_confusion_matrix(y_test >= .5, predictions >= .5, class_names, save_path='./training/cm/%s_cm_%s_%s.png' % (current_model, dt, batch_size))
