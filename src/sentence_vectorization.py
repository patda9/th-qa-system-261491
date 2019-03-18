import json
import numpy as np
import os
import preprocessing as prep
import re

np.random.seed(0)

def m_words_separate(m, arrays_of_tokens, overlapping_words=0, question_number=0):
    sentences_in_articles = []
    sentences_ranges_in_articles = []
    for i in range(arrays_of_tokens.__len__()):
        sentences = []
        sentences_ranges = []
        temp_j = 0
        for j in range(0, arrays_of_tokens[i].__len__(), m-overlapping_words):
            if((j + m) > arrays_of_tokens[i].__len__()):
                if(arrays_of_tokens[i].__len__() < m):
                    fill_length = m - arrays_of_tokens[i].__len__()
                    print('-fill length', fill_length, arrays_of_tokens[i].__len__())
                    arrays_of_tokens[i] = list(arrays_of_tokens[i])
                    for k in range(fill_length):
                        arrays_of_tokens[i].append(arrays_of_tokens[i][-1])
                    arrays_of_tokens[i] = np.asarray(arrays_of_tokens[i])
                    idx = (j, m)
                    sentence = arrays_of_tokens[i][j:m]
                    sentences_ranges.append(idx)
                else:
                    fill_length = (j + m) - arrays_of_tokens[i].__len__()
                    idx = (j-fill_length, arrays_of_tokens[i].__len__())
                    sentence = arrays_of_tokens[i][j-fill_length:j+m]
                    sentences_ranges.append(idx)
            else:
                if(j > 0):
                    idx = (temp_j - overlapping_words, (temp_j + m) - overlapping_words)
                    sentences_ranges.append(idx)
                    temp_j += (m - overlapping_words)
                else:
                    idx = (temp_j, temp_j + m)
                    sentences_ranges.append(idx)
                    temp_j += m
                sentence = arrays_of_tokens[i][j:j+m]
            sentences.append(sentence)
        sentences_in_articles.append(sentences)
        sentences_ranges_in_articles.append(sentences_ranges)
        # print('Batch: ' + str(question_number + 1) + ' Converting to ' + str(m) + '-words sentences. [' + str(i) + '/' + str(arrays_of_tokens.__len__()) + '] \r', end='')
        # print('\n')
    return [np.asarray(sentences_in_articles), np.asarray(sentences_ranges_in_articles)]

if(__name__ == '__main__'):
    path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/'
    dataset = os.listdir(path)

    n = 512
    sample_ids = []
    for i in range(n):
        randomed_doc_id = int(dataset[np.random.randint(dataset.__len__())].split('.')[0])
        while(randomed_doc_id in sample_ids):
            randomed_doc_id = int(dataset[np.random.randint(dataset.__len__())].split('.')[0])
        sample_ids.append(randomed_doc_id)

    count = 1
    samples = []
    for article_id in sample_ids:
        text_file_path = os.path.join(path + str(article_id) + '.json')
        with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as text_file:
            data = json.load(text_file)
        samples.append(data)
        count += 1

    for i in range(samples.__len__()):
        samples[i] = prep.remove_xml(samples[i])
    vocabularies = [w for doc in samples for w in doc]
    vocabularies = prep.remove_noise(vocabularies)[0]

    ## word to id transformation
    word2id = {}
    word2id['<PAD>'] = 0
    word2id['<NIV>'] = 1
    for (i, w) in enumerate(set(vocabularies), 2):
        try:
            word2id[w] = i
        except ValueError:
            word2id[w] = 1

    id2word = {idx: w for w, idx in word2id.items()}
    sample2id = {article_id: i for i, article_id in enumerate(list(sample_ids))}
    print((sample2id))

    ## words to word ids representation
    # arrays_of_tokens = []
    # for i in range(samples.__len__()):
    #     arrays_of_tokens.append([word2id[w] for w in samples[i]])

    # remapped_article_ids = []
    # for i in range(sample_ids.__len__()):
    #     remapped_article_ids.append(sample2id[sample_ids[i]])

    words_per_sentence = 30
    overlapping_words = words_per_sentence // 2
    overlap_flag = True
    # article_sentences = m_words_separate(words_per_sentence, arrays_of_tokens, overlapping_words=overlapping_words)

    # from keras.preprocessing import sequence

    # padded_article_sentences = []
    # for i in range(article_sentences.__len__()):
    #     padded_s = sequence.pad_sequences(article_sentences[i], maxlen=words_per_sentence)
    #     padded_article_sentences.append(padded_s)

    # sentences_article_ids = []
    # y_train_category = []
    # for i in range(padded_article_sentences.__len__()):
    #     y_train_category.append(remapped_article_ids[i])
    #     for j in range(padded_article_sentences[i].__len__()):
    #         sentences_article_ids.append(remapped_article_ids[i])

    # flatten_article_sentences = np.vstack(padded_article_sentences) # label each sentence with its article id

    # from keras.utils import to_categorical

    # x = flatten_article_sentences.copy()
    # y = sentences_article_ids.copy()
    # input_output_label = list(zip(x, y))
    # np.random.shuffle(input_output_label)
    # x, y = list(zip(*input_output_label))
    # x = np.asarray(list(x))
    # y = np.asarray(list(y))

    # x_train = x[:int(.8 * x.__len__())]
    # y_train = y[:int(.8 * y.__len__())]
    # x_test = x[int(.8 * x.__len__()):]
    # y_test = y[int(.8 * y.__len__()):]

    # print(x_test)

    from keras.layers import Activation, Bidirectional, Dense, Embedding, Flatten, InputLayer, LSTM
    from keras.layers.core import Masking
    from keras.models import load_model, Model, Sequential
    from keras.optimizers import Adam
    from keras.utils import to_categorical

    embedding_size = 64
    lstm_output_size = 128

    model = Sequential()
    model.add(Embedding(word2id.__len__(), embedding_size, input_length=words_per_sentence))
    model.add(Masking(mask_value=0, input_shape=(words_per_sentence, 1))) # word id is only a feature
    model.add(Bidirectional(LSTM(lstm_output_size)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(sample2id.__len__(), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    # model.fit(x_train, to_categorical(y_train, num_classes=sample2id.__len__()), batch_size=32, epochs=8)

    # if(overlap_flag):
    #     model.save('./' + str(words_per_sentence) + 'w-' + str(overlapping_words) + '-overlap-sentence-vectorization-model-' + str(n) + '.h5')
    # else:
    #     model.save('./' + str(words_per_sentence) + 'w-sentence-vectorization-model-' + str(n) + '.h5')

    # scores = model.evaluate(x_test, to_categorical(y_test, num_classes=sample2id.__len__()))
    # print(scores)
    # print(f"{model.metrics_names[1]}: {scores[1] * 100} %")

    # dense1_layer = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
    # dense1_layer_output = dense1_layer.predict(np.asarray(x_test))
    # print('sentence vectorization output vvv ( dimension:', dense1_layer_output[0].__len__(), ')')
    # print(dense1_layer.predict(np.asarray(x_test)))
    # softmax_layer = Model(inputs=model.input, outputs=model.get_layer(index=4).output)
    # softmax_layer_output = softmax_layer.predict(np.asarray(x_test))
    # print(x_test)
    # print('softmax classification probabilities output vvv ( classes:', softmax_layer_output[0].__len__(), ')')
    # print(softmax_layer.predict(np.asarray(x_test)))
