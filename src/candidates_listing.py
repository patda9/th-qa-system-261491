import json
import numpy as np

def vectorize_words(question_candidates, word_vectors, embedding_shape=(100, )):
    embedded_sentences = []
    for i in range(question_candidates.__len__()):
        temp_article = []
        for j in range(question_candidates[i].__len__()):
            temp_sentence = []
            for k in range(question_candidates[i][j].__len__()):
                try:
                    embedded_token = word_vectors[question_candidates[i][j][k]]
                    temp_sentence.append(embedded_token)
                except KeyError:
                    temp_sentence.append(np.zeros(embedding_shape))
            temp_article.append(np.asarray(temp_sentence))
        embedded_sentences.append(np.asarray(temp_article))
    return embedded_sentences

def vectorize_questions(tokenized_questions, word_vectors, embedding_shape=(100, ), words_per_sentence=20):
    embedded_questions = []
    for i in range(tokenized_questions.__len__()):
        temp_embedded_questions = []
        for j in range(tokenized_questions[i].__len__()):
            try:
                embedded_token = word_vectors[tokenized_questions[i][j]]
                temp_embedded_questions.append(embedded_token)
            except:
                temp_embedded_questions.append(np.zeros(embedding_shape))
        while(temp_embedded_questions.__len__() < words_per_sentence):
            temp_embedded_questions.insert(0, np.zeros(embedding_shape))
        while(temp_embedded_questions.__len__() > words_per_sentence):
            temp_embedded_questions.pop()
        embedded_questions.append(np.asarray(temp_embedded_questions))
    return embedded_questions

def calculate_distance(candidate_sentence_vectors, question_vectors):
    distance_matrix = []
    for i in range(candidate_sentence_vectors.__len__()):
        temp = []
        for j in range(candidate_sentence_vectors[i].__len__()):
            temp.append(np.linalg.norm(candidate_sentence_vectors[i][j] - question_vectors[i]))
        distance_matrix.append(temp)
    return distance_matrix

def locate_plain_text_characters(m_word_sentence_ranges, min_distance_indexes, original_tokens_ranges):
    plain_text_character_positions = []
    sentence_indexes = []
    for i in range(min_distance_indexes.__len__()):
        temp_all_sentences = []
        temp = []
        for j in range(min_distance_indexes[i].__len__()):
            temp_one_sentence = []
            min_dist_idx = min_distance_indexes[i][j]
            sentence_range = m_word_sentence_ranges[i][min_dist_idx] # tuple of candidate sentence range
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

def sort_distances(distance_matrix, max_num_candidate=25):
    min_distance_indexes = []
    ordered_distance_matrix = []
    for i in range(distance_matrix.__len__()):
        # argsort()[:len(distance_matrix[i])] => ascending order ranking from 0 (or 1) to len(distance_matrix[i])
        if(distance_matrix[i].__len__() < 25):
            min_index = np.asarray(distance_matrix[i]).argsort()[:distance_matrix[i].__len__()]
            sorted_dist = np.sort(distance_matrix[i])[:distance_matrix[i].__len__()]
        else:
            min_index = np.asarray(distance_matrix[i]).argsort()[:max_num_candidate]
            sorted_dist = np.sort(distance_matrix[i])[:max_num_candidate]
        ordered_distance_matrix.append(sorted_dist)
        min_distance_indexes.append(min_index)
    
    min_distance_indexes = np.asarray(min_distance_indexes)
    return min_distance_indexes, ordered_distance_matrix

def locate_candidates_sentence(embedded_questions, embedded_sentences, m_word_sentence_ranges, 
                                original_tokens_ranges, dense_layer):
    for i in range(embedded_sentences.__len__()):
        if(not(embedded_sentences[i].size)):
            print('fucking problem:', i)
            embedded_sentences[i] = embedded_questions[i-1]
    
    candidate_sentence_vectors = []
    for i in range(embedded_sentences.__len__()):
        try:
            candidate_sentence_vectors.append(dense_layer.predict(embedded_sentences[i]))
        except:
            print(embedded_sentences[i])
            print(i)

    distance_matrix = calculate_distance(candidate_sentence_vectors, embedded_questions)
    min_distance_indexes, min_distance_matrix = sort_distances(distance_matrix)
    plaint_text_character_positions, sentence_indexes = locate_plain_text_characters(m_word_sentence_ranges, min_distance_indexes, original_tokens_ranges)

    return plaint_text_character_positions, min_distance_matrix, sentence_indexes

if(__name__ == '__main__'):
    dist = np.array([
                    [100, 200, 150],
                    [128, 64, 32],
                    [2, 2, 1]
                    ])