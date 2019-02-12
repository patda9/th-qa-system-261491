
import findDocuments as fd
import findAnswer as fa
import json
import numpy as np
import preprocessing as prep
import os
import re
import sentence_vectorization as sv
import candidates_listing as cl

from gensim.models import Word2Vec
from keras.models import load_model, Model
from pprint import pprint

if(__name__ == '__main__'):
    # """
    # <findDocuments()>
    # """
    # candidate_output = fd.findDocuments()
    # with open('./result/find_documents_output.json', 'w', errors='ignore') as find_doc:
    #     json.dump(candidate_output, find_doc)

    # """
    # sentence and question comparison scripts
    # output: (begin, end) characters positions <json file>
    # input: candidate document ids <json file>, questions
    # """
    # print('Loading sentence vector model...')
    # vectorization_model = load_model('./models/20w-10-overlap-sentence-vectorization-model-768-16.h5')
    # vectorization_model.summary()
    # dense_layer = Model(input=vectorization_model.input, output=vectorization_model.get_layer(index=5).output)
    # print('Use dense layer to generate sentence vectors.')

    # with open('./result/find_documents_output.json', 'r') as ids:
    #     candidate_document_ids = json.load(ids)

    # tokens_path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/'

    # WORDS_PER_SENTENCE = 20
    # OVERLAPPING_WORDS = WORDS_PER_SENTENCE // 2

    # # preprocess tokens
    # candidate_documents = [] # [i] for each question
    # original_token_indexes = []
    # original_token_lengths = []
    # for i in range(candidate_document_ids.__len__()):
    #     dot = '.'
    #     print('Processing question [' + str(i) + '/' + str(candidate_document_ids.__len__()) + '] candidate documents. \r', end='')
    #     indexes = []
    #     lengths = []
    #     tokens = []
    #     for j in range(candidate_document_ids[i].__len__()):
    #         article_path = os.path.join(tokens_path, str(candidate_document_ids[i][j]) + '.json')
    #         with open(article_path, 'r', encoding='utf-8', errors='ignore') as doc:
    #             document = json.load(doc)
    #             preprocessed_document = prep.remove_noise(document)
    #         indexes.append(preprocessed_document[1])
    #         lengths.append(preprocessed_document[2])
    #         tokens.append(preprocessed_document[0])
    #     candidate_documents.append(tokens)
    #     original_token_indexes.append(indexes)
    #     original_token_lengths.append(lengths)

    # print('Merge tokens in document into 20-words sentences.')
    # m_words_preprocessed_documents = []
    # m_words_sentence_ranges = []
    # for i in range(candidate_documents.__len__()):
    #     m_words_preprocessing = sv.m_words_separate(WORDS_PER_SENTENCE, candidate_documents[i], overlapping_words=OVERLAPPING_WORDS, question_number=i)
    #     m_words_docs = np.asarray(m_words_preprocessing[0])
    #     m_words_ranges = np.asarray(m_words_preprocessing[1])
    #     if(m_words_docs.__len__()):
    #         m_words_preprocessed_documents.append(m_words_docs)
    #     m_words_sentence_ranges.append(m_words_ranges)

    # wv_model = Word2Vec.load('C:/Users/Patdanai/Desktop/492/word2vec.model')
    # word_vectors = wv_model.wv

    # MAX_NUMBER_OF_WORDS = word_vectors.vocab.__len__()
    # MAX_SEQUENCE_LENGHT = WORDS_PER_SENTENCE

    # EMBEDDING_SHAPE = word_vectors['มกราคม'].shape # use as word vector's dimension

    # # questions vectorization goes here
    # questions = []
    # # change path to tokenized questions
    # with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
    #     questions = json.load(f)

    # tokenized_questions = []
    # for i in range(questions.__len__()):
    #     questions[i] = [w for w in questions[i] if not(w is ' ')]
    #     tokenized_questions.append(questions[i])

    # print('Vectorizing input questions 0.')
    # embedded_questions = cl.vectorize_questions(tokenized_questions, word_vectors, embedding_shape=EMBEDDING_SHAPE, words_per_sentence=WORDS_PER_SENTENCE)
    # embedded_questions = np.asarray(embedded_questions)
    # print('Vectorizing input questions 1.')
    # question_vectors = dense_layer.predict(embedded_questions)

    # question_idx = 0
    # candidate_answers = []
    # candidate_character_positions = []
    # distance_scores = []
    # for question_candidates in m_words_preprocessed_documents: # for each question
    #     print('question id:', question_idx)
    #     embedded_candidate = cl.vectorize_words(question_candidates, word_vectors, embedding_shape=EMBEDDING_SHAPE)
    #     character_positions, min_distance_matrix, sentence_indexes = cl.locate_candidates_sentence(question_vectors[question_idx], embedded_candidate, m_words_sentence_ranges[question_idx], 
    #                                                                                                 original_token_lengths[question_idx], dense_layer)
    #     candidate_for_each_doc = []
    #     for i in range(character_positions.__len__()): # each candidate doc
    #         sentence_candidates = []
    #         for j in range(character_positions[i].__len__()): # each candidate sentence
    #             begin_position = character_positions[i][j][0]
    #             end_position = character_positions[i][j][-1]
    #             begin_index = sentence_indexes[i][j][0]
    #             end_index = sentence_indexes[i][j][1]
    #             score = min_distance_matrix[i][j]
    #             candidate = {
    #                 "question_id": question_idx + 1,
    #                 "sentence": candidate_documents[question_idx][i][begin_index:end_index], 
    #                 "answer_begin_position ": begin_position,
    #                 "answer_end_position": end_position,
    #                 "article_id": candidate_document_ids[question_idx][i],
    #                 "similarity_score": float(score)
    #             }
    #             sentence_candidates.append(candidate)
    #         candidate_for_each_doc.append(sentence_candidates)
    #     candidate_answers.append(candidate_for_each_doc)
    #     question_idx += 1

    # with open('./result/candidate_sentences.json', 'w', encoding='utf-8') as cand:
    #     try:
    #         json.dump(candidate_answers, cand, ensure_ascii=False, indent=4)
    #     except:
    #         json.dump(candidate_answers, cand, indent=4)

    a = json.load(open('./../new_sample_questions.json', encoding='utf-8-sig'))
    a = a['data']
    question = json.load(open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8-sig'))

    question_index = []
    doc_id = []
    real_answer = []
    question_type = [
        ['กี่', 'ปี ใด', 'ปี อะไร', 'พ.ศ.  อะไร', 'ค.ศ.  อะไร', 'พ.ศ. อะไร', 'ค.ศ. อะไร', 'พ.ศ. ใด', 'พ.ศ.  ใด', 'ค.ศ. ใด',
        'ค.ศ.  ใด', 'เท่า ไร', 'เท่า ไหร่', 'เท่า ใด', 'คริสต์ศักราช ใด', 'จำนวน ใด']
        , ['เมื่อ ไร', 'เวลา ใด', 'วัน ใด', 'เมื่อ ใด', 'วัน ที่']  # date format
        , ['ใคร', 'ว่า อะไร', 'ชื่อ อะไร', 'คน ใด', 'คน ไหน', 'คือใคร', 'ผู้ ใด']  # human name
        , ['ประเทศ ใด', 'ประเทศ อะไร']
        , ['จังหวัดใด', 'จังหวัด ใด', 'จังหวัด อะไร']
        , ['เมืองใด', 'เมือง ใด', 'เมือง อะไร']
        , ['ภาค ใด']
        , ['แคว้น ใด']
        , ['ทวีปใด', 'ทวีป อะไร', 'ทวีป ใด', 'ภูมิภาค ไหน']
        , ['ที่ ไหน', 'ที่ ใด']  # where
        , ['อะไร', 'อย่าง ไร', 'ใด', 'ไหน']  # other what, other dai, other nhai
    ]
    thai_number_text = [u'หนึ่ง', u'สอง', u'สาม', u'สี่', u'ห้า', u'หก', u'เจ็ด', u'แปด', u'เก้า', u'สิบ', u'สิบเอ็ด']
    month = ['มกราคม', 'ม.ค.',
            'กุมภาพันธ์', 'ก.พ.',
            'มีนาคม', 'มี.ค.',
            'เมษายน', 'เม.ย.',
            'พฤษภาคม', 'พ.ค.',
            'มิถุนายน', 'มิ.ย.',
            'กรกฎาคม', 'ก.ค.',
            'สิงหาคม', 'ส.ค.',
            'กันยายน', 'ก.ย.',
            'ตุลาคม', 'ต.ค.',
            'พฤศจิกายน', 'พ.ย.',
            ' ธันวาคม', 'ธ.ค.',
            'พ.ศ.','ค.ศ.']

    class_label = [2,3,4,5,6,7,8,9,10]
    word_class = [[],[]]
    for i in class_label:
        tmp = json.load(open("./data/word_class/" + str(i) + ".json", "r", encoding="utf-8"))
        word_class.append(set(tmp))
    wrong = 0
    all_rs = []
    possible_answer = []
    answer_position = []
    answer_json = []

    inp = json.load(open("./result/candidate_sentences.json", "r", encoding="utf-8"))
    for i in range(inp.__len__()):
        inp[i] = inp[i][0]

    for i in range(wrong, inp.__len__()):
        answer = a[i]['answer']
        # inp.append(make_sentence_answer(article_id, answer_begin))  ### input
        real_answer.append(answer)
        s = ''.join(question[i])
        possible_answer.append([])
        doc_id.append([])
        answer_position.append([])

        rr_score = []
        for l in range(question_type.__len__()):
            if any(fa.check_question_type(k, question[i]) for k in question_type[l]):
                question_word_index = fa.find_question_word(question[i], question_type[l])
                for j in inp[i]:
                    doc_id[-1].append([j["article_id"]])
                    possible_answer[-1].append([])
                    if l > 1:
                        possible_answer[-1][-1], doc_id[-1][-1] = fa.find_candidate(possible_answer[-1][-1], doc_id[-1][-1], j['sentence'], l, word_class)
                        rr_score.append(fa.find_answer_word(j,j['answer_begin_position ']))
                    elif l == 0:
                        for k in range(j['sentence'].__len__()):
                            if fa.hasNumbers(j['sentence'][k]):
                                doc_id[-1][-1].append(k)
                                possible_answer[-1][-1].append(j['sentence'][k])
                        if possible_answer[-1][-1].__len__() < 1 :
                            possible_answer[-1].pop()
                            doc_id[-1].pop()
                            continue
                        rr_score.append(fa.find_answer_word(j,j['answer_begin_position ']))
                    else:
                        for k in range(j['sentence'].__len__()):
                            if j['sentence'][k] in month:
                                doc_id[-1][-1].append(k)
                                possible_answer[-1][-1].append(j['sentence'][k])
                            else:
                                possible_answer[-1][-1], doc_id[-1][-1] = fa.find_candidate(possible_answer[-1][-1], doc_id[-1][-1],j['sentence'], l)
                        rr_score.append(fa.find_answer_word(j,j['answer_begin_position ']))
                break

            elif l == 10 and not any(fa.check_question_type(k, question[i]) for k in question_type[l]):
                tmp_q = []
                for q in question[i]:
                    tmp = []
                    for w in question_type[l]:
                        tmp.append(fa.similar(q, w))
                    tmp_q.append([question[i].index(q), max(tmp)])
                tmp_q.sort(key=lambda s: s[1], reverse=True)
                question_word_index = [tmp_q[0][0], question[i][tmp_q[0][0]]]
                for j in inp[i]:
                    doc_id[-1].append([j["article_id"]])
                    possible_answer[-1].append([])
                    print("\n#############################\n")
                    possible_answer[-1][-1], doc_id[-1][-1] = fa.find_candidate(possible_answer[-1][-1], doc_id[-1][-1], j['sentence'], l)
                    rr_score.append(fa.find_answer_word(j,j['answer_begin_position ']))

        print(doc_id[i][rr_score.index(max(rr_score))])

        for_answer_json = {}
        for_answer_json['question_id'] = i+1
        for_answer_json['question'] = s
        for_answer_json['answer'] = doc_id[i][rr_score.index(max(rr_score))][1]
        for_answer_json['answer_begin_position '] = answer_position[i][rr_score.index(max(rr_score))][0] #### w8 input
        for_answer_json['answer_end_position'] = answer_position[i][rr_score.index(max(rr_score))][1] #### w8 input
        for_answer_json['article_id'] = doc_id[i][rr_score.index(max(rr_score))][1] #### w8 input
        answer_json.append(for_answer_json)

    with open('./result/output_answer.json', 'w' , encoding="utf-8") as outfile:
        json.dump(answer_json,outfile, indent=4, ensure_ascii=False)
