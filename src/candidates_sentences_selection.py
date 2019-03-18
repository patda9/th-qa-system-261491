import candidates_listing as cl
import findDocuments as fd
import json
import numpy as np
import os
import preprocessing as prep
import rnn_compare_twotext as rc

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, LSTM, Multiply, Subtract, Dropout, GRU, Masking, Concatenate
from pprint import pprint

rnn_size = 32
sentence_length = 40
word_vector_length = 100

def get_original_token_positions(document_id, documents_path):
    doc_path = os.path.join(documents_path, str(document_id) + '.json')
    with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
        document = json.load(f)
    preprocessed_document = prep.remove_noise(document)
    doc_index = list(range(len(document)))
    
    return doc_index, preprocessed_document[2]

def get_sentence_vectorization_layer(model, idx=5):
    from keras.models import Model

    vectorization_layer = Model(input=model.input, output=model.get_layer(index=idx).output)
    
    return vectorization_layer

def load_json_doc_ids(path='D:/Users/Patdanai/Workspace/th-qa-system-261491/data/'):
    with open(path, 'r') as f:
        candidate_ids = json.load(f)
    
    return candidate_ids
    
def load_corpus_word_vectors(path='D:/Users/Patdanai/th-qasys-db/word_vectors_model/word2vec.model'):
    from gensim.models import Word2Vec
    wv_model = Word2Vec.load(path)
    return wv_model.wv

def load_document_word_vectors(document_ids, wv_path='D:/Users/Patdanai/th-qasys-db/preprocessed_corpus_wv'):
    return np.load(wv_path + str(document_ids) + '.npy')

def load_sentence_vectorization_model(model_path):
    from keras.models import load_model

    model = load_model(model_path)
    model.summary()
    
    return model

def load_tokenized_questions(path):
    with open(path, 'r') as f:
        questions = json.load(f)
    
    return questions, len(questions)

def calculate_distance(vectorized_question, vectorized_sentences):
    distance_matrices = []
    for i in range(vectorized_sentences.__len__()):
        distance_matrix = np.zeros((vectorized_sentences[i].shape[0], ))
        for j in range(vectorized_sentences[i].__len__()):
            distance_matrix[j] = np.linalg.norm(candidate_sentence_vectors[i][j] - vectorized_question)
        distance_matrices.append(distance_matrix)

    return distance_matrices

def sort_distances(distance_matrices, max_num_candidate=8):
    min_distance_indexes = []
    ordered_distance_matrices = []
    for i in range(distance_matrices.__len__()):
        # argsort()[:len(distance_matrix[i])] => ascending order ranking from 0 (or 1) to len(distance_matrix[i])
        if(distance_matrices[i].__len__() < max_num_candidate):
            min_index = np.asarray(distance_matrices[i]).argsort()[:distance_matrices[i].__len__()]
            sorted_dist = np.sort(distance_matrices[i])[:distance_matrices[i].__len__()]
        else:
            min_index = np.asarray(distance_matrices[i]).argsort()[:max_num_candidate]
            sorted_dist = np.sort(distance_matrices[i])[:max_num_candidate]
        ordered_distance_matrices.append(sorted_dist)
        min_distance_indexes.append(min_index)
    
    min_distance_indexes = np.asarray(min_distance_indexes)
    return min_distance_indexes, ordered_distance_matrices

def locate_plain_text_characters(sentence_ranges, min_distance_indexes, original_tokens_ranges):
    plain_text_character_positions = []
    sentence_indexes = []
    for i in range(min_distance_indexes.__len__()):
        temp_all_sentences = []
        temp = []
        for j in range(min_distance_indexes[i].__len__()):
            temp_one_sentence = []
            min_dist_idx = min_distance_indexes[i][j]
            sentence_range = sentence_ranges[i][min_dist_idx] # tuple of candidate sentence range
            for k in range(sentence_range[0], sentence_range[1]):
                try:
                    character_position = original_tokens_ranges[i][k]
                except:
                    character_position = original_tokens_ranges[i][-1]
                temp_one_sentence.append(character_position)
            temp_all_sentences.append(temp_one_sentence)
            temp.append(sentence_range)
        plain_text_character_positions.append(temp_all_sentences)
        sentence_indexes.append(temp)
    return plain_text_character_positions, sentence_indexes

def locate_candidate_answers(vectorized_question, vectorized_sentences, sentence_ranges, 
                                original_tokens_ranges, max_num_candidate=8):
    distance_matrix = calculate_distance(vectorized_question, vectorized_sentences)
    min_distance_indexes, min_distance_matrix = sort_distances(distance_matrix, max_num_candidate=max_num_candidate)
    plaint_text_character_positions, sentence_indexes = locate_plain_text_characters(sentence_ranges, min_distance_indexes, original_tokens_ranges)

    return plaint_text_character_positions, min_distance_matrix, sentence_indexes

def vectorize_question_tokens(tokenized_question, word_vectors, embedded_question=[], embedding_shape=(100, ), words_per_sentence=20):
    for j in range(tokenized_question.__len__()): # for word in tokenized question
        try:
            embedded_token = word_vectors[tokenized_question[j]]
            embedded_question.append(embedded_token)
        except:
            embedded_question.append(np.zeros(embedding_shape))
    while(embedded_question.__len__() < words_per_sentence):
        embedded_question.insert(0, np.zeros(embedding_shape))
    while(embedded_question.__len__() > words_per_sentence):
        embedded_question = embedded_question[:words_per_sentence]

    return np.asarray(embedded_question)

# models structure
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

def candidate_similarity(candidate_document_ids=None): # None for file reading
    DOCUMENTS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'
    MODEL_PATH = './compare2text/compare_model_v5.h5'
    SV_PATH = 'D:/Users/Patdanai/th-qasys-db/corpus_sv/'
    WV_PATH = 'D:/Users/Patdanai/th-qasys-db/corpus_wv/'
    WV_MODEL_PATH = 'D:/Users/Patdanai/th-qasys-db/word_vectors_model/word2vec.model'
    questions_path = './results/question_sentence_tokens/question_sentence_tokens.json'
    questions_path = 'C:\\Users\\Patdanai\\Workspace\\th-qa-system-261491\\results\\final\\question_sentence_tokens.json'

    # MODEL_PATH = './models/compare_model_v3/191-0.7933.h5'
    # questions_path = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/data/ThaiQACorpus-EvaluationDataset-tokenize.json'
    # questions_path = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/data/new_sample_questions_tokenize.json'
    
    WORDS_PER_SENTENCE = 20
    OVERLAPPING_WORDS = WORDS_PER_SENTENCE // 2

    word_vectors = load_corpus_word_vectors(path=WV_MODEL_PATH)

    gru_sv = Input(shape=(rnn_size, ), name='pass_candidate_sv') # create tensor: sentence vector(s) database

    qsv_model = sentenceVector() # submodel
    qsv_model.load_weights('./compare2text/compare_model_v5.h5', by_name=True) # load submodel weight
    qsv_model.summary()
    qs_seq = Input(shape=(sentence_length, word_vector_length)) # create tensor
    qsv_model = qsv_model(qs_seq) # add input tensor: question sequence

    sc_model = sentenceCompare()
    sc_model.load_weights('./compare2text/compare_model_v5.h5', by_name=True)
    sc_model.summary()
    similarity = sc_model([gru_sv, qsv_model]) # add input tensors: candidate sv, question sv

    # form model
    model = Model(inputs=[gru_sv, qs_seq], outputs=similarity)
    model.load_weights(MODEL_PATH)
    model.summary()

    tokenized_questions, questions_num = load_tokenized_questions(questions_path) # use this question num

    begin_question = 0
    with open('./results/final/candidate_doc_ids.json', 'r') as f:
        candidate_document_ids = json.load(f)

    copy = []
    candidate_answers = []
    part = 0
    chr_pointer = 0
    print('Part: %d' % part)
    # implement small batch processing (1 question/batch)
    for i in range(0, candidate_document_ids.__len__()): # question
        print('Processing question [' + str(i) + '/' + str(candidate_document_ids.__len__()) + '] candidate documents. \r')
        array_of_wvs = []
        documents_index = [] # original one
        documents_lengths = [] # original one
        tokenized_docs = []

        temp_length = 7
        if(len(candidate_document_ids[i]) < temp_length):
            temp_length = len(candidate_document_ids[i])

        for j in range(0, temp_length): # candidate doc
        # for j in range(0, candidate_document_ids[i].__len__()): # candidate doc
            original_index, original_lengths = get_original_token_positions(candidate_document_ids[i][j], DOCUMENTS_PATH)
            array_of_wvs.append(np.load(SV_PATH + 'sv-' + str(candidate_document_ids[i][j]) + '.npy'))
            documents_index.append(original_index)
            documents_lengths.append(original_lengths)
            tokenized_docs.append(load_document_word_vectors(candidate_document_ids[i][j], WV_PATH))
        
        m_words = prep.m_words_separate(WORDS_PER_SENTENCE, tokenized_docs, overlapping_words=WORDS_PER_SENTENCE//2)
        m_words_sentences = m_words[0]
        m_words_index_ranges = m_words[1]

        question_wvs = []
        temp = []
        for k in range(len(tokenized_questions[i]['sentence_tokens'])):
            try:
                temp.append(word_vectors[tokenized_questions[i]['sentence_tokens'][k]])
            except:
                temp.append(np.zeros((rc.word_vector_length, )))
        question_wvs.append(temp)
        question_wvs = np.array(question_wvs)
        
        temp = np.zeros((question_wvs.shape[0], rc.sentence_length - question_wvs.shape[1], rc.word_vector_length))
        temp[:] = 0.
        question_wvs = np.concatenate((temp, question_wvs), axis=1)

        ordered_similarity = []
        c_index_ranges = []
        for j in range(len(array_of_wvs)):
            temp = np.repeat(question_wvs, array_of_wvs[j].shape[0], axis=0)
            prediction = model.predict([array_of_wvs[j], temp])

            ranks = np.argsort(prediction, axis=0).flatten()

            temp0 = []
            temp1 = []
            for k in range(len(ranks)):
                temp0.append(prediction.flatten()[ranks[k]])
                temp1.append(m_words_index_ranges[j][ranks[k]])
            ordered_similarity.append(temp0)
            c_index_ranges.append(temp1)
        
        sentences_index = []
        for j in range(len(m_words_index_ranges)):
            temp = []
            for k in range(len(c_index_ranges[j])):
                start = c_index_ranges[j][k][0]
                end = c_index_ranges[j][k][1]
                temp.append(documents_index[j][start:end])
            sentences_index.append(temp)

        plain_text_character_positions = []
        for j in range(len(sentences_index)):
            temp_k = []
            for k in range(len(sentences_index[j])):
                temp_l = []
                for l in range(len(sentences_index[j][k])):
                    idx = sentences_index[j][k][l]
                    temp_l.append(documents_lengths[j][idx])
                temp_k.append(temp_l)
            plain_text_character_positions.append(temp_k)
        
        temp_j = []
        for j in range(0, temp_length):
        # for j in range(len(candidate_document_ids[i])):
            with open((DOCUMENTS_PATH + str(candidate_document_ids[i][j]) + '.json'), 
                        'r', encoding='utf-8', 
                            errors='ignore') as f:
                document_content = json.load(f)
            
            temp_k = []
            for k in range(len(plain_text_character_positions[j])):
                begin_position = plain_text_character_positions[j][k][0]
                end_position = plain_text_character_positions[j][k][-1]
                begin_index = sentences_index[j][k][0]
                end_index = sentences_index[j][k][-1]
                score = ordered_similarity[j][k]
                try:
                    candidate = {
                        "question_id": i + 1,
                        "article_id": candidate_document_ids[i][j], 
                        "candidate_rank": len(plain_text_character_positions[j]) - k - 1,  
                        "sentence": document_content[begin_index:end_index], 
                        "answer_begin_position ": begin_position, 
                        "answer_end_position": end_position, 
                        "similarity_score": float(score)
                    }
                except IndexError:
                    candidate = {
                        "question_id": i + 1,
                        "article_id": candidate_document_ids[i][j], 
                        "candidate_rank": len(plain_text_character_positions[j]) - k - 1, 
                        "sentence": document_content[begin_index:], 
                        "answer_begin_position ": begin_position, 
                        "answer_end_position": end_position, 
                        "similarity_score": float(score)
                    }
                # print(candidate)
                temp_k.append(candidate)
            temp_j.append(temp_k)
        candidate_answers.append(temp_j)
        copy.append(temp_j)

        if((i+1) % int(questions_num * .05) == 0 or i == len(candidate_document_ids)):
            if(part > 9):
                no = chr(65 + chr_pointer)
                chr_pointer += 1
                print(no)
            else:
                no = part
                print(no)
            out_path = './results/final/candidate_sentences/candidate_sentences_part' + str(no) + '.json'
            with open(out_path, 'w', 
                        encoding='utf-8', errors='ignore') as f:
                json.dump(candidate_answers, f, indent=4)
            candidate_answers = []
            part += 1

    return copy
