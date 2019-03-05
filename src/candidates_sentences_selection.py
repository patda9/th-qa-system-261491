import candidates_listing as cl
import findDocuments as fd
import json
import numpy as np
import os
import preprocessing as prep
import sentence_vectorization as sv

from pprint import pprint

def get_original_token_positions(document_id, documents_path):
    doc_path = os.path.join(documents_path, str(document_id) + '.json')
    with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
        document = json.load(f)
        preprocessed_document = prep.remove_noise(document)
    
    return preprocessed_document[1], preprocessed_document[2]

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
    # for i in range(tokenized_questions.__len__()):
    for j in range(tokenized_question.__len__()): # for word in tokenized question
        try:
            embedded_token = word_vectors[tokenized_question[j]]
            embedded_question.append(embedded_token)
        except:
            embedded_question.append(np.zeros(embedding_shape))
    while(embedded_question.__len__() < words_per_sentence):
        embedded_question.insert(0, np.zeros(embedding_shape))
        print(embedded_question.__len__())
    while(embedded_question.__len__() > words_per_sentence):
        embedded_question = embedded_question[:words_per_sentence]

    return np.asarray(embedded_question)

if __name__ == "__main__":
    DOCUMENTS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'
    SV_MODEL_PATH = 'D:/Users/Patdanai/th-qasys-db/sentence_vectorization_models/20w-10-overlap-sentence-vectorization-model-768-16.h5'
    SV_MODEL_PATH = './models/20wps-2000samples-16epochs-0-02.h5'
    WV_PATH = 'D:/Users/Patdanai/th-qasys-db/preprocessed_corpus_wv/'
    WV_MODEL_PATH = 'D:/Users/Patdanai/th-qasys-db/word_vectors_model/word2vec.model'
    
    # question_path = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/data/ThaiQACorpus-EvaluationDataset-tokenize.json'
    question_path = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/data/new_sample_questions_tokenize.json'
    
    WORDS_PER_SENTENCE = 20
    OVERLAPPING_WORDS = WORDS_PER_SENTENCE // 2

    word_vectors = load_corpus_word_vectors(path=WV_MODEL_PATH)
    sv_model = load_sentence_vectorization_model(SV_MODEL_PATH)
    sv_layer = get_sentence_vectorization_layer(sv_model)

    tokenized_questions, questions_num = load_tokenized_questions(question_path) # use this question num
    print(questions_num)
    # questions_num =

    # candidate_document_ids = [['115035', '229360', '544013', '722958', '70381', '884481'], ['376583', '736347', '562166', '804147', '3817', '52640'], 
    #                             ['115035', '229360', '544013', '722958', '70381', '884481'], ['376583', '736347', '562166', '804147', '3817', '52640']]

    begin_question = 0

    # candidate_document_ids = fd.findDocuments(begin_question, questions_num)
    # print(len(candidate_document_ids))

    with open('./data/output_findDOC4000.json', 'r') as f:
        candidate_document_ids = json.load(f)

    candidate_answers = []
    part = 0
    print('part:', part)

    # implement small batch processing (1 question/batch)
    for i in range(0, candidate_document_ids.__len__()): # question
        # print('Processing question [' + str(i) + '/' + str(candidate_document_ids.__len__()) + '] candidate documents. \r', end='')
        
        documents_index = [] # original one
        documents_lengths = [] # original one
        array_of_wvs = []
        for j in range(candidate_document_ids[i].__len__()): # candidate doc
            original_index, original_lengths = get_original_token_positions(candidate_document_ids[i][j], DOCUMENTS_PATH)
            array_of_wvs.append(load_document_word_vectors(candidate_document_ids[i][j], WV_PATH))
            documents_index.append(original_index)
            documents_lengths.append(original_lengths)
        # print(len(array_of_wvs))
        # print(len(documents_index))
        # print(len(documents_lengths))

        # make sample becomes batch (size 1) for feeding through sv_layer
        # embedded_question = np.expand_dims(vectorize_question_tokens(tokenized_questions[i], word_vectors), axis=0) # question_tokens => [question_tokens]
        # question_vector = sv_layer.predict(embedded_question).flatten()
        # print(embedded_question.shape)
        # print(question_vector.shape)

        m_tokens_groupping = sv.m_words_separate(WORDS_PER_SENTENCE, array_of_wvs, overlapping_words=OVERLAPPING_WORDS, question_number=i)
        m_tokens_documents = np.asarray(m_tokens_groupping[0])
        m_tokens_ranges = np.asarray(m_tokens_groupping[1])
        
        # print(m_tokens_documents)
        # for i in m_tokens_documents:
        #     print('i', i.__len__())
        #     for j in i:
        #         print('j', j.__len__())
        
        # print(m_tokens_ranges.__len__())

        candidate_sentence_vectors = []
        for j in range(len(m_tokens_documents)):
            candidate_sentence_vectors.append(sv_layer.predict(np.asarray(m_tokens_documents[j])))
        # print(candidate_sentence_vectors[1][0].shape)
        candidate_sentence_vectors = np.asarray(candidate_sentence_vectors)

        ### sentence, question vectorization
        # print(tokenized_questions[i])
        embedded_question = vectorize_question_tokens(tokenized_questions[i], word_vectors)
        # print(embedded_question.shape)
        vectorized_question = sv_layer.predict(np.array([embedded_question]))
        # print(vectorized_question.flatten().shape)

        distance_matrices = calculate_distance(vectorized_question, candidate_sentence_vectors)
        # print(distance_matrices)
        min_distance_indexes, ordered_distance_matrices = sort_distances(distance_matrices)
        # print(min_distance_indexes)
        # print(ordered_distance_matrices)
        plain_text_character_positions, sentence_indexes = locate_plain_text_characters(m_tokens_ranges, 
                                                                                        min_distance_indexes, 
                                                                                        documents_lengths)
        # print(plain_text_character_positions)
        # print(sentence_indexes)
        plain_text_character_positions, min_distance_matrix, sentence_indexes = locate_candidate_answers(vectorized_question, 
                                                                                                            candidate_sentence_vectors, 
                                                                                                            m_tokens_ranges, 
                                                                                                            documents_lengths, max_num_candidate=16)
        
        # print(min_distance_matrix)
        # print(len(min_distance_matrix))
        # print('---', len(plain_text_character_positions))
        
        temp_j = []
        for j in range(len(candidate_document_ids[i])):
            
            with open((DOCUMENTS_PATH + str(candidate_document_ids[i][j]) + '.json'), 
                        'r', encoding='utf-8', 
                            errors='ignore') as f:
                document_content = json.load(f)
            
            # print(len(candidate_document_ids[i]))
            # print(plain_text_character_positions[i].__len__())
            
            temp_k = []
            for k in range(len(plain_text_character_positions[j])):
                begin_position = plain_text_character_positions[j][k][0]
                end_position = plain_text_character_positions[j][k][-1]
                begin_index = sentence_indexes[j][k][0]
                end_index = sentence_indexes[j][k][1]
                score = min_distance_matrix[j][k]
                candidate = {
                    "question_id": i + 1,
                    "article_id": candidate_document_ids[i][j], 
                    "candidate_no": k + 1, 
                    # "sentence": document_content[begin_index:end_index], # uncomment this line for winner's output
                    "answer_begin_position ": begin_position,
                    "answer_end_position": end_position,
                    "similarity_score": float(score)
                }
                # temp_k.append(candidate)
                # pprint(candidate)
                temp_k.append(candidate)
            temp_j.append(temp_k)
        candidate_answers.append(temp_j)
        # print(candidate_answers)
        # print('i', i)
        # if((i+1) % int(questions_num * .1) == 0 or i == len(candidate_document_ids)):
 
        if((i+1) % int(questions_num * .1) == 0 or i == len(candidate_document_ids)):
            with open('./results/candidate_answers_part' + str(part) + '.json', 'w', 
                        encoding='utf-8', errors='ignore') as f:
                json.dump(candidate_answers, f, indent=4)
            candidate_answers = []
            part += 1
    print('\n')

# plot histogram of true answer from json output
    # foreach nsc_answer_detail
        # get answer token ranges
        # locate answer index(token range)
        # check if candidate answer sentence range is overlap with nsc answer token range
    # plot histogram 
