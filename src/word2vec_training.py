import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import preprocessing as prep
import re
import sentence_vectorization as sv
import os

from gensim.models import Word2Vec
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from pprint import pprint

np.random.seed(0)

if(__name__ == '__main__'):
    tokens_path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/' # get tokenized articles content
    plain_text_path = 'C:/Users/Patdanai/Desktop/documents-nsc/' # get plain text article content
    tokens_dataset = os.listdir(tokens_path)
    n_samples = 512+256 # number of samples from nsc questions

    models_ws_archs_path = './models'
    model_files = os.listdir(models_ws_archs_path)

    nsc_sample_questions = []
    with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
        nsc_sample_questions = json.load(f)
    
    nsc_answer_doc_id = []
    with open('./../new_sample_questions_answer.json', 'r', encoding='utf-8', errors='ignore') as f:
        nsc_answer_doc_id = json.load(f)

    nsc_answer_details = {}
    with open('./../new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        nsc_answer_details = json.load(f)
    
    # get first n samples from questions
    count = 0
    last_doc_id = 282972
    selected_article_ids = []
    selected_questions_numbers = []
    selected_plain_text_questions = []
    for q in nsc_answer_details['data']:
        if(q['article_id'] < last_doc_id and count < n_samples): # limitted preprocessed docs id: 282972
            selected_article_ids.append(q['article_id'])
            selected_questions_numbers.append(q['question_id'])
            selected_plain_text_questions.append(q['question'])
            count += 1

    # print(np.asarray(list(zip(selected_questions_numbers, selected_article_ids)))) # TESTING FUNCTION: map question numbers to article ids

    """
    output: array of tokens <arrays of tokenized article content: array like>
    input: path of tokens <file path: string>
    """
    # load tokens from each article
    selected_plain_text_article = []
    selected_tokenized_articles = [] # array for arrays of tokens
    for i in selected_article_ids:
        article_path = os.path.join(tokens_path, str(i) + '.json')
        plain_text_article_path = os.path.join(plain_text_path, str(i) + '.txt')
        with open(plain_text_article_path, 'r', encoding='utf-8', errors='ignore') as f:
            plain_text_article = f.read()
        selected_plain_text_article.append(plain_text_article)
        with open(article_path, 'r', encoding='utf-8', errors='ignore') as f:
            article_tokens = json.load(f)
        selected_tokenized_articles.append(article_tokens)

    # print(np.asarray(selected_tokenized_articles)) # TESTING FUNCTION: arrays of article tokens

    """
    output: preprocessed questions tokens <arrays of tokenized questions: array like>
    input: questions tokens <arrays of tokenized questions: array like>
    """
    # remove noise from tokenized questions
    selected_tokenized_questions = []
    for i in range(nsc_sample_questions.__len__()):
        if((i+1) in selected_questions_numbers): # use (i+1) as index: first question starts at 1 
            nsc_sample_questions[i] = [w for w in nsc_sample_questions[i] if not(w is ' ')]
            selected_tokenized_questions.append(nsc_sample_questions[i])

    # print(np.asarray(selected_tokenized_questions)) # TESTING FUNCTION: arrays of questions tokens
    
    remaining_tokens = []
    original_tokens_indexes = []
    original_tokens_ranges = []
    j = 1
    for ids in selected_article_ids:
        file_path = os.path.join(tokens_path, str(ids) + '.json')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            preprocessed_article = prep.remove_noise(data)
        remaining_tokens.append(preprocessed_article[0])
        original_tokens_indexes.append(preprocessed_article[1])
        original_tokens_ranges.append(preprocessed_article[2])
        j += 1 

    # print(np.asarray(remaining_tokens[0])) # TESTING FUNCTION: arrays of preprocessed tokens,
    # and also => preprocessed tokens indexes
    # and also => and preprocessed tokens ranges
    # print(np.asarray(original_tokens_indexes[0]))
    # print(np.asarray(list(zip(original_tokens_indexes[0], remaining_tokens[0])))) # TESTING FUNCTION: arrays of preprocessed tokens
    # and preprocessed tokens indexes
    # print(np.asarray(original_tokens_ranges[0])) # each before-preprocessing token's ending position

    # create vocabularies from input articles
    vocabularies = [article[i] for article in remaining_tokens for i in range(article.__len__())]
    # create word to id dictionary
    word2id = {}
    for (i, w) in enumerate(set(vocabularies)):
        try:
            word2id[w] = i
        except ValueError:
            pass
    # create word_id to word dictionary
    id2word = {idx: w for w, idx in word2id.items()}

    doc_id2class = {doc_id: i for i, doc_id in enumerate(list(selected_article_ids))} # ****
    
    # pprint(word2id) # TESTING FUNCTION: dict of words: ids
    # pprint(id2word) # TESTING FUNCTION: dict of ids: words

    """
    output: 
    input: 
    """
    # preprocess questions
    preprocessed_questions = []
    for question in selected_tokenized_questions:
        temp = prep.remove_noise(question)[0]
        preprocessed_questions.append(temp)
    
    # print(np.asarray(preprocessed_questions))

    words_per_sentence = 20
    overlapping_words = words_per_sentence // 2

    m_words_preprocessed = sv.m_words_separate(words_per_sentence, remaining_tokens, overlapping_words=overlapping_words)
    m_words_preprocessed_article = m_words_preprocessed[0]
    m_words_preprocessed_sentence_ranges = m_words_preprocessed[1]

    # print(np.asarray(m_words_preprocessed_article))
    # print(np.asarray(m_words_preprocessed_sentence_ranges))

    saved_model = Word2Vec.load('C:/Users/Patdanai/Desktop/492/word2vec.model')
    # print(saved_model)
    word_vectors = saved_model.wv
    # print("Example of word vectors: {}".format(word_vectors.vocab['วนิดา']))

    max_number_of_words = word_vectors.vocab.__len__()
    max_sequence_length = words_per_sentence
    # print(max_number_of_words)

    vocabularies = [article[i] for article in remaining_tokens for i in range(article.__len__())]
    word_index = {token: i+1 for i, token in enumerate(set(vocabularies))}
    index_word = {i+1: token for i, token in enumerate(set(vocabularies))}

    # print(word_index)
    # print(index_word)
    
    embedding_shape = word_vectors['มกราคม'].shape # use as word vector's dimension
    embedded_sentences = [] # use as x (input) of network
    document_ids = [] # use as output classes
    for i in range(m_words_preprocessed_article.__len__()):
        temp_article = []
        temp_document_id = []
        for j in range(m_words_preprocessed_article[i].__len__()):
            temp_sentence = []
            for k in range(m_words_preprocessed_article[i][j].__len__()):
                try:
                    embedded_token = word_vectors[m_words_preprocessed_article[i][j][k]]
                    temp_sentence.append(embedded_token)
                except:
                    temp_sentence.append(np.zeros(embedding_shape))
            
            temp_article.append(np.asarray(temp_sentence))
            temp_document_id.append(selected_article_ids[i])
        document_ids.append(temp_document_id)
        embedded_sentences.append(np.asarray(temp_article))

    print(embedded_sentences[0][0].shape)
    print(embedded_sentences[0][0][0].shape)

    for i in range(embedded_sentences.__len__()):
        for j in range(embedded_sentences[i].__len__()):
            if(embedded_sentences[i][j].shape != (20, 100)):
                print('i:', i, 'j:', j, 'shape[1] != 20')
            for k in range(embedded_sentences[i][j].__len__()):
                try:
                    if(embedded_sentences[i][j][k].shape != (100, )):
                        print('i:', i, 'j:', j, 'k:', k, 'shape[2] != 100')
                except:
                    print('i:', i, 'j:', j, 'k:', k)
                    print(embedded_sentences[i][j])

    # exit()

    from keras.utils import to_categorical

    flatten_embedded_sentences = np.vstack(embedded_sentences)
    flatten_document_ids = np.hstack(document_ids)
    yx_tuple = list(zip(flatten_document_ids, flatten_embedded_sentences))
    np.random.shuffle(yx_tuple)
    flatten_document_ids, flatten_embedded_sentences = list(zip(*yx_tuple))

    x = np.asarray(flatten_embedded_sentences)
    x_train = x[:int(.8 * x.__len__())]
    x_test = x[int(.8 * x.__len__()):]

    flatten_document_ids = np.asarray(flatten_document_ids)
    document_classes = []
    for doc_id in flatten_document_ids:
        document_classes.append(doc_id2class[doc_id])
    
    y = to_categorical(document_classes, dtype=np.int32)
    y_train = y[:int(.8 * y.__len__())]
    y_test = y[int(.8 * y.__len__()):]

    from keras.layers import Activation, Bidirectional, Dense, Embedding, Flatten, Input, LSTM, Masking, SpatialDropout1D, BatchNormalization
    from keras.models import load_model, Model
    from keras.optimizers import Adam

    epochs = 16
    lstm_output_size = 128
    
    # input layer
    embedded_sequences = Input(shape=(max_sequence_length, embedding_shape[0]))
    masked_sequences = (Masking(mask_value=0.))(embedded_sequences)
    dropout_input_sequences = SpatialDropout1D(.25)(masked_sequences)

    # lstm network layer
    lstm_layer = Bidirectional(LSTM(lstm_output_size))(dropout_input_sequences)
    
    # output layer
    batch_norm_sequences = BatchNormalization()(lstm_layer)
    word_vector_weights = Dense(embedding_shape[0])(batch_norm_sequences)
    predictions = Dense(selected_article_ids.__len__(), activation='softmax')(word_vector_weights)

    # construct model
    model = Model(inputs=embedded_sequences, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(epsilon=1e-8), metrics=['accuracy'])
    model.summary()

    print(doc_id2class)
    model.fit(x_train, y_train, batch_size=1, epochs=epochs)
    
    # TODO next prepare training set (x_train, y_train) testing set (x_test, y_test) 
    # *!* random more sample from corpus that not in 4000 nsc qustions    
    model.save('./models/' + str(words_per_sentence) + 'w-' + str(overlapping_words) + '-overlap-sentence-vectorization-model-' + str(n_samples) + str(epochs) + '.h5')

    scores = model.evaluate(x_test, y_test)
    print(f"{model.metrics_names[1]}: {scores[1] * 100} %")

    # dense1_layer = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
    # dense1_layer_output = dense1_layer.predict(np.asarray(x_test))
    # print('sentence vectorization output vvv ( dimension:', dense1_layer_output[0].__len__(), ')')
    # print(dense1_layer.predict(np.asarray(x_test)))
    # softmax_layer = Model(inputs=model.input, outputs=model.get_layer(index=4).output)
    # softmax_layer_output = softmax_layer.predict(np.asarray(x_test))
    # print(x_test)
    # print('softmax classification probabilities output vvv ( classes:', softmax_layer_output[0].__len__(), ')')
    # print(softmax_layer.predict(np.asarray(x_test)))
