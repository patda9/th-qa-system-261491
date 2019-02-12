import json
import numpy as np
import preprocessing as prep
import os
import re
import sentence_vectorization as sv
import candidates_listing as cl

from gensim.models import Word2Vec
from keras.layers import Activation, Bidirectional, Dense, Input, LSTM, SpatialDropout1D, BatchNormalization
from keras.models import load_model, Model
from keras.optimizers import Adam
from pprint import pprint

vectorization_model = load_model('./models/20w-10-overlap-sentence-vectorization-model-768-16.h5')
vectorization_model.summary()
dense_layer = Model(input=vectorization_model.input, output=vectorization_model.get_layer(index=5).output)

# 1. find doc => [docid1..docidn]
# 2. load tokened doc from corpus => [doc1..docn]
# 3. preprocess tokened doc
#   => remaining tokens
#   => origin index
#   => origin character position
# 4. doc => classes
# 5. preprocess tokened question
# 6. 20 word per sentence
# 

# tokened docs, tokened questions

with open('./result/test_out.json', 'r') as ids:
    candidate_document_ids = json.load(ids)

candidate_document_ids = candidate_document_ids[:4]

tokens_path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/'

# candidate docs => 
# { q1: [id1, id2, ...], 
# q2: [id1, id2, ...], 
# ...}

candidate_documents = [] # [i] for each question
original_token_indexes = []
original_token_lengths = []
for i in range(candidate_document_ids.__len__()):
    indexes = []
    lengths = []
    tokens = []
    for j in range(candidate_document_ids[i].__len__()):
        article_path = os.path.join(tokens_path, str(candidate_document_ids[i][j]) + '.json')
        with open(article_path, 'r', encoding='utf-8', errors='ignore') as doc:
            document = json.load(doc)
            preprocessed_document = prep.remove_noise(document)
        indexes.append(preprocessed_document[1])
        lengths.append(preprocessed_document[2])
        tokens.append(preprocessed_document[0])
    candidate_documents.append(tokens)
    original_token_indexes.append(indexes)
    original_token_lengths.append(lengths)

# print(candidate_documents[0].__len__())
# print(original_token_indexes)
# print(original_token_lengths)

WORDS_PER_SENTENCE = 20
OVERLAPPING_WORDS = WORDS_PER_SENTENCE // 2

m_words_preprocessed_documents = []
m_words_sentence_ranges = []
for i in range(candidate_documents.__len__()):
    m_words_preprocessing = sv.m_words_separate(WORDS_PER_SENTENCE, candidate_documents[i], overlapping_words=OVERLAPPING_WORDS)
    m_words_docs = np.asarray(m_words_preprocessing[0])
    m_words_ranges = np.asarray(m_words_preprocessing[1])
    m_words_preprocessed_documents.append(m_words_docs)
    m_words_sentence_ranges.append(m_words_ranges)

# print(m_words_preprocessed_documents[0])
# print(m_words_sentence_ranges[0])

wv_model = Word2Vec.load('C:/Users/Patdanai/Desktop/492/word2vec.model')
word_vectors = wv_model.wv

MAX_NUMBER_OF_WORDS = word_vectors.vocab.__len__()
MAX_SEQUENCE_LENGHT = WORDS_PER_SENTENCE

EMBEDDING_SHAPE = word_vectors['มกราคม'].shape # use as word vector's dimension

# questions vectorization goes here
#   => now fill question with np.zeros(EMBEDDING_SHAPE) for testing only
embedded_questions = []
for i in range(candidate_documents.__len__()):
    temp = []
    for j in range(WORDS_PER_SENTENCE):
        temp.append(np.zeros(EMBEDDING_SHAPE))
    embedded_questions.append(temp)

nsc_sample_questions = []
with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
    nsc_sample_questions = json.load(f)

tokenized_questions = []
for i in range(nsc_sample_questions.__len__()):
    nsc_sample_questions[i] = [w for w in nsc_sample_questions[i] if not(w is ' ')]
    tokenized_questions.append(nsc_sample_questions[i])

embedded_questions = cl.vectorize_questions(tokenized_questions, word_vectors, embedding_shape=EMBEDDING_SHAPE, words_per_sentence=WORDS_PER_SENTENCE)

embedded_questions = np.asarray(embedded_questions)
question_vectors = dense_layer.predict(embedded_questions)
exit()

print(m_words_preprocessed_documents[0][0].__len__())
question_idx = 0

candidate_answers = []
candidate_character_positions = []
distance_scores = []
for question_candidates in m_words_preprocessed_documents: # for each question
    embedded_candidate = cl.vectorize_words(question_candidates, word_vectors, embedding_shape=EMBEDDING_SHAPE)
    character_positions, min_distance_matrix = cl.locate_candidates_sentence(question_vectors[question_idx], embedded_candidate, m_words_sentence_ranges[question_idx], 
                                                        original_token_lengths[question_idx], dense_layer)
    
    print(character_positions.__len__())
    print(min_distance_matrix.__len__())

    candidate_for_each_doc = []
    for i in range(character_positions.__len__()): # each candidate doc
        sentence_candidates = []
        for j in range(character_positions[i].__len__()): # each candidate sentence
            begin_position = character_positions[i][j][0]
            end_position = character_positions[i][j][-1]
            score = min_distance_matrix[i][j]
            candidate = {
                "question_id": question_idx + 1,
                "answer_begin_position ": begin_position,
                "answer_end_position": end_position,
                "article_id": candidate_document_ids[question_idx][i],
                "similarity_score": float(score)
            }
            sentence_candidates.append(candidate)
        candidate_for_each_doc.append(sentence_candidates)
    candidate_answers.append(candidate_for_each_doc)

    # distance_scores.append(min_distance_matrix)

    # for i in range(character_position.__len__()):
    #     for j in range(character_position[i].__len__()):
    #         # temp = []
    #         begin_position = character_position[i][j]
    #         end_position = character_position[i][j]
    #         # temp.append((begin_position, end_position))
    #         candidate = {
    #             "question_id": question_idx + 1,
    #             "answer_begin_position ": begin_position,
    #             "answer_end_position": end_position,
    #             "article_id": candidate_document_ids[question_idx][i],
    #             "similarity_score": min_distance_matrix[question_idx][i]
    #         }
    #     candidate_answers.append(candidate)

    # print(candidate_answers)

    # answer_details = {
    #     "rank": 0,
    #     "question_id": 1,
    #     "question": "สุนัขตัวแรกรับบทเป็นเบนจี้ในภาพยนตร์เรื่อง Benji ที่ออกฉายในปี พ.ศ. 2517 มีชื่อว่าอะไร",
    #     "answer_begin_position ": 468,
    #     "answer_end_position": 527,
    #     "article_id": 115035,
    #     "similarity_score": 32.90196990966797
    # }
    
    question_idx += 1

with open('./result/candidates_out.json', 'w') as cand:
    json.dump(candidate_answers, cand)