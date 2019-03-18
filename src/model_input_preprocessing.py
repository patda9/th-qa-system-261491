import json
import numpy as np
import os
import preprocessing as prep
import sentence_vectorization as sv

from pprint import pprint

# def get_original_token_positions(document_id, documents_path):
#     doc_path = os.path.join(documents_path, str(document_id) + '.json')
#     with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
#         document = json.load(f)
#         preprocessed_document = prep.remove_noise(document)
    
#     return preprocessed_document[1], preprocessed_document[2]

def load_corpus_word_vectors(path='D:/Users/Patdanai/th-qasys-db/word_vectors_model/word2vec.model'):
    from gensim.models import Word2Vec
    wv_model = Word2Vec.load(path)
    return wv_model.wv

def load_tokenized_questions(path):
    with open(path, 'r') as f:
        questions = json.load(f)
    return questions, len(questions)

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
    with open('./data/output_findDOC4000.json', 'r') as f:
        candidate_document_ids = json.load(f)

    DOCUMENTS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'
    WV_PATH = 'D:/Users/Patdanai/th-qasys-db/preprocessed_corpus_wv/'
    WV_MODEL_PATH = 'D:/Users/Patdanai/th-qasys-db/word_vectors_model/word2vec.model'

    questions_path = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/data/new_sample_questions_tokenize.json'
    # tokenized_questions, questions_num = load_tokenized_questions(questions_path) # use this question num

    # print(questions_num)
    # print(np.array(tokenized_questions[0:32]))

    # word_vectors = load_corpus_word_vectors(WV_MODEL_PATH)
    # vectorized_q = vectorize_question_tokens(tokenized_questions, word_vectors)

    ## -------------------------------- get sentence index ranges -------------------------------- ##
    #################################################################################################

    nsc_answer_details = {}
    with open('./data/new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        nsc_answer_details = json.load(f)

    # n_samples = len(nsc_answer_details['data']) - 3600
    n_samples = len(nsc_answer_details['data'])

    count = 0
    selected_article_ids = []
    selected_questions_numbers = []
    selected_plain_text_questions = []
    for q in nsc_answer_details['data']:
        if(count < n_samples): # limitted preprocessed docs id: 282972
            selected_article_ids.append(q['article_id'])
            selected_questions_numbers.append(q['question_id'])
            selected_plain_text_questions.append(q['question'])
            count += 1

    print(np.array(selected_article_ids))
    print(n_samples)

    # with open('./data/article_ids.json', 'w') as f:
        # json.dump(selected_article_ids, f)

    ### ---------------- get preprocessed documents ---------------- ###
    tokens_path = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'

    original_tokens = []
    original_tokens_indexes = []
    original_tokens_ranges = []
    remaining_tokens = []
    j = 1
    for ids in selected_article_ids:
        file_path = os.path.join(tokens_path, str(ids) + '.json')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            original_tokens.append(data)
            preprocessed_article = prep.remove_noise(data)
        remaining_tokens.append(preprocessed_article[0])
        # original_tokens_indexes.append(preprocessed_article[1])
        original_tokens_ranges.append(preprocessed_article[2])
        j += 1

    # print('Documents:', len(remaining_tokens))
    # print('Tokens index:', len(original_tokens_indexes))
    # print('Tokens lengths:', len(original_tokens_ranges))

    # # words_per_sentence = 20
    # # overlapping_words = words_per_sentence // 2

    # # m_words_preprocessing = sv.m_words_separate(words_per_sentence, remaining_tokens, overlapping_words=overlapping_words)
    # # m_words_preprocessed_articles = np.asarray(m_words_preprocessing[0])
    # # m_words_sentence_ranges = np.asarray(m_words_preprocessing[1])

    # # print(len(m_words_sentence_ranges))

    # # exit()

    # ### ---------------- get preprocessed documents ---------------- ###

    # ### ---------------- get answer characters positions ---------------- ###
    # answer_char_positions = []
    # for i in range(selected_questions_numbers.__len__()):
    #     for q in nsc_answer_details['data']:
    #         if(selected_questions_numbers[i] == q['question_id']):
    #             answer_begin_pos = q['answer_begin_position '] - 1 # pos - 1 to refer array index
    #             answer_end_pos = q['answer_end_position'] - 1 # pos - 1 to refer array index
    #             answer_char_positions.append((answer_begin_pos, answer_end_pos))
    
    # print('Answer chracters positions:', len(answer_char_positions))
    # ### ---------------- get answer characters positions ---------------- ###

    # ### ---------------- get answer index range by its token begin, end positions ---------------- ###
    # answer_indexes = []
    # original_indexes = []
    # for i in range(remaining_tokens.__len__()):
    #     temp_original_index = []
    #     temp_answer_index = []
    #     for j in range(original_tokens_ranges[i].__len__()):
    #         begin = answer_char_positions[i][0] # + 1 to turn back to original position
    #         end = answer_char_positions[i][1] + 1 # range => end -1
    #         eca = original_tokens_ranges[i][j] # ending character of the answer
    #         if(eca in range(begin, end)):
    #             temp_original_index.append(j+1)
    #     for j in range(remaining_tokens[i].__len__()):
    #         print((original_tokens_indexes[i][j]))
    #         if(original_tokens_indexes[i][j] in temp_original_index):
    #             temp_answer_index.append(j)
    #             original_indexes.append(temp_original_index)
    #             break
    #         if(j+1 == remaining_tokens[i].__len__()):
    #             original_indexes.append([])
    #     try:
    #         answer_indexes.append(temp_answer_index)
    #         original_indexes[i].pop() # (temp_prep_ans_idx_range[0], temp_prep_ans_idx_range[-1] + 1), (begin, end)))
    #     except IndexError:
    #         pass
    
    # print('Answers index:', len(answer_indexes))
    # print('Origin:', len(original_tokens_indexes))
    
    # with open('./data/candidate_doc_index.json', 'w') as f:
    #     json.dump(answer_indexes, f)
    
    # with open('./data/original_answers_index.json', 'w') as f:
    #     json.dump(original_indexes, f)
    
    # # print(preprocessed_article)
    
    # ### ---------------- get answer index range by its token begin, end positions ---------------- ###

    # #################################################################################################
    # ## -------------------------------- get sentence index ranges -------------------------------- ##

    # answers_char_positions = []
    # for i in range(len(os.listdir('./results/tmp/'))):
    #     with open('./results/tmp/candidate_answers_part%d.json' % (i), 'r') as f:
    #         data = json.load(f)
    #         for j in range(len(data)):
    #             temp_k = [] # candidate from question id
    #             for k in range(len(data[j])):
    #                 temp = [] # candidates from document id
    #                 for candidate in data[j][k]:
    #                     temp.append((candidate['answer_begin_position '], candidate['answer_end_position']))
    #                 try:
    #                     temp_k.append(temp)
    #                 except:
    #                     temp_k.append(temp[:len(temp)])
    #             answers_char_positions.append(temp_k)
    # print(len(answers_char_positions))

    # with open('./data/tmp/answers_char_positions.json', 'w') as f:
    #     json.dump(answers_char_positions, f)

    # exit()
    # nsc_answer_details = {}
    # with open('./data/new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        # nsc_answer_details = json.load(f)

    with open('./data/output_findDOC4000.json') as f:
        candidate_doc_ids = json.load(f)

    with open('./data/tmp/answers_char_positions.json', 'r') as f:
        questions_data = json.load(f)

    candidate_ans_char_pos = []
    for i in range(len(questions_data)): # char pos by question
        temp = []
        for j in range(len(questions_data[i])): # char positions by candidate doc
            temp.append(questions_data[i][j])
        candidate_ans_char_pos.append(temp)
    
    print(len(candidate_doc_ids))
    print(len(questions_data))

    nsc_answer_details = nsc_answer_details['data']

    correct_sentences = []
    incorrect_sentences = []
    for i in range(len(nsc_answer_details)): # question
        print('question[%d]' % i)
        ans_begin_pos = nsc_answer_details[i]['answer_begin_position ']
        ans_end_pos = nsc_answer_details[i]['answer_end_position']
        
        c_doc_index = 0
        inc_doc = []
        for c_doc in candidate_doc_ids[i]:
            if(str(nsc_answer_details[i]['article_id']) == str(c_doc)):
                # print(i, c_doc, c_doc_index)
                # print(len(candidate_ans_char_pos[i][c_doc_index]))

                for j in range(len(candidate_ans_char_pos[i][c_doc_index])):
                    token_index = 0
                    tokens_count = 1
                    temp_c = []
                    temp_inc = []
                    start, end = candidate_ans_char_pos[i][c_doc_index][j][0], \
                                candidate_ans_char_pos[i][c_doc_index][j][1] + 1
                    
                    for word in original_tokens_ranges[i]:
                        # print(word, start, end)
                        if(word in range(start, end)):
                            if(ans_begin_pos in range(start, end) and ans_end_pos in range(start, end)):
                                temp_c.append(token_index)
                            else:
                                temp_inc.append(token_index)
                            tokens_count += 1
                        token_index += 1
                    if(temp_c):
                        correct_sentences.append({
                                "article_id": c_doc, 
                                "question_id": (i),  
                                "sentence_index": temp_c
                            })
                    if(temp_inc):
                        inc_doc.append({ 
                                "article_id": c_doc, 
                                "question_id": (i), 
                                "rank": (j + 1), 
                                "sentence_index": temp_inc
                            })
                incorrect_sentences.append(inc_doc)
            c_doc_index += 1

    with open('./results/correct_sentences_index/correct_sentences_index.json', 'w') as f:
        json.dump(correct_sentences, f)

    with open('./results/incorrect_sentences_index/incorrect_sentences_index.json', 'w') as f:
        json.dump(incorrect_sentences, f)























                # for j in range(len(candidate_ans_char_pos[i])):
                #     for k in range(len(candidate_ans_char_pos[i][j])):
                #         token_index = 0
                #         tokens_count = 1

                #         temp_correct = []
                #         temp_incorrect = []
                        
                #         cs_begin_pos = candidate_ans_char_pos[i][j][k][0]
                #         cs_end_pos = candidate_ans_char_pos[i][j][k][1]
                #         for word in original_tokens_ranges[i]:
                #             if(word in range(cs_begin_pos, cs_end_pos + 1)):
                #                 if(ans_begin_pos in range(cs_begin_pos, cs_end_pos + 1) \
                #                     and ans_end_pos in range(cs_begin_pos, cs_end_pos + 1)):
                #                     temp_correct.append(token_index)
            # else:
                # for word in original_tokens_ranges[i]:
                    # if( in range(cs_begin_pos, cs_end_pos + 1)):
                        # temp_correct.append()
            # print(temp_correct)
                # correct_s.append()



    
    
    
    
    
    
    
    
    #     candidate_doc_index_inc = []
    #     candidate_doc_index_c = []
    #     for j in range(len(candidate_ans_char_pos[i])):
    #         sentence_tokens_index_c = []
    #         sentence_tokens_index_inc = []
    #         for k in range(len(candidate_ans_char_pos[i][j])):
    #             token_index = 0
    #             tokens_count = 1
    #             temp_c = []
    #             temp_inc = []
    #             for word in original_tokens_ranges[i]:
    #                 if(word in range(candidate_ans_char_pos[i][j][k][0], candidate_ans_char_pos[i][j][k][1] + 1)):
    #                     ans_begin_pos = nsc_answer_details['data'][i]['answer_begin_position ']
    #                     ans_end_pos = nsc_answer_details['data'][i]['answer_end_position']
    #                     for c_doc in candidate_doc_ids[i]:
    #                         # print(c_doc)
    #                         if(str(c_doc) == str(nsc_answer_details['data'][i]['article_id'])):
    #                             if(ans_begin_pos in range(candidate_ans_char_pos[i][j][k][0], candidate_ans_char_pos[i][j][k][1] + 1) \
    #                                 and ans_end_pos in range(candidate_ans_char_pos[i][j][k][0], candidate_ans_char_pos[i][j][k][1] + 1)):
    #                                 temp_c.append(token_index)
    #                                 # print(1)
    #                             else:
    #                                 temp_inc.append(token_index)
    #                             break
    #                     # print('index: {0}, part of sentence: {1}, token char range: ({2}, {3}), tokens count: {4}'.format(token_index, word, candidate_ans_char_pos[i][j][k][0], candidate_ans_char_pos[i][j][k][1], tokens_count))
    #                     # tokens_count += 1
    #                 token_index += 1
    #             sentence_tokens_index_c.append(temp_c)
    #             sentence_tokens_index_inc.append(temp_inc)
    #             # print(len(sentence_tokens_index))
    #         candidate_doc_index_c.append(sentence_tokens_index_c)
    #         candidate_doc_index_inc.append(sentence_tokens_index_inc)
    #         # print(len(candidate_doc_index))
    #     with open('./data/tmp/tmp1/correct/correct_index_q%d.json' % (i+1), 'w') as f:
    #         json.dump(candidate_doc_index_c, f)
    #     with open('./data/tmp/tmp1/incorrect/incorrect_index_q%d.json' % (i+1), 'w') as f:
    #         json.dump(candidate_doc_index_inc, f)
        