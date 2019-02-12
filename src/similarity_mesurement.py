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
    n_samples = 32 # number of samples from nsc questions

    models_ws_archs_path = 'C:/Users/Patdanai/Desktop/492/th-qa-system-261491/models'
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

    print(np.asarray(remaining_tokens[0])) # TESTING FUNCTION: arrays of preprocessed tokens,
    # and also => preprocessed tokens indexes
    # and also => and preprocessed tokens ranges
    print(np.asarray(original_tokens_indexes[0]))
    print(np.asarray(list(zip(original_tokens_indexes[0], remaining_tokens[0])))) # TESTING FUNCTION: arrays of preprocessed tokens
    # and preprocessed tokens indexes
    print(np.asarray(original_tokens_ranges[0])) # each before-preprocessing token's ending position

    # create vocabularies from input articles
    vocabularies = [article[i] for article in remaining_tokens for i in range(article.__len__())]
    # create word to id dictionary
    word2id = {}
    for (i, token) in enumerate(set(vocabularies)):
        try:
            word2id[token] = i
        except ValueError:
            pass
    # create word_id to word dictionary
    id2word = {idx: w for w, idx in word2id.items()}
    
    doc_id2class = {doc_id: i for i, doc_id in enumerate(list(selected_article_ids))}

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
    
    print(np.asarray(preprocessed_questions))

    words_per_sentence = 20
    overlapping_words = words_per_sentence // 2

    m_words_preprocessing = sv.m_words_separate(words_per_sentence, remaining_tokens, overlapping_words=overlapping_words)
    m_words_preprocessed_articles = np.asarray(m_words_preprocessing[0])
    m_words_sentence_ranges = np.asarray(m_words_preprocessing[1])

    saved_wv_model = Word2Vec.load('C:/Users/Patdanai/Desktop/492/word2vec.model')
    word_vectors = saved_wv_model.wv
    print("Example of word vectors: {}".format(word_vectors.vocab['มกราคม']))

    max_number_of_words = word_vectors.vocab.__len__()
    max_sequence_length = words_per_sentence

    embedding_shape = word_vectors['มกราคม'].shape # use as word vector's dimension
    embedded_sentences = [] # use as x (input) of network
    document_ids = [] # use as output classes
    for i in range(m_words_preprocessed_articles.__len__()):
        temp_article = []
        temp_document_id = []
        for j in range(m_words_preprocessed_articles[i].__len__()):
            temp_sentence = []
            for k in range(m_words_preprocessed_articles[i][j].__len__()):
                try:
                    embedded_token = word_vectors[m_words_preprocessed_articles[i][j][k]]
                    temp_sentence.append(embedded_token)
                except:
                    temp_sentence.append(np.zeros(embedding_shape))
            temp_article.append(np.asarray(temp_sentence))
            temp_document_id.append(selected_article_ids[i])
        document_ids.append(temp_document_id)
        embedded_sentences.append(np.asarray(temp_article))
    
    print(embedded_sentences[-1])
    print(document_ids[-1])

    # find ans index
    answer_char_positions = []
    for i in range(selected_questions_numbers.__len__()):
        for q in nsc_answer_details['data']:
            if(selected_questions_numbers[i] == q['question_id']):
                answer_begin_pos = q['answer_begin_position '] - 1 # pos - 1 to refer array index
                answer_end_pos = q['answer_end_position'] - 1 # pos - 1 to refer array index
                answer_char_positions.append((answer_begin_pos, answer_end_pos))

    print(answer_char_positions)

    answers = []
    for i in range(n_samples):
        begin = answer_char_positions[i][0]
        end = answer_char_positions[i][-1]
        answer = selected_plain_text_article[i][begin:end]
        answers.append(answer)
    
    print(answers)

    answer_indexes = []
    for i in range(remaining_tokens.__len__()):
        temp_original_index = []
        temp_answer_index = []
        for j in range(original_tokens_ranges[i].__len__()):
            begin = answer_char_positions[i][0] # + 1 to turn back to original position
            end = answer_char_positions[i][1] + 1 # range => end -1
            eca = original_tokens_ranges[i][j] # ending character of the answer
            if(eca in range(begin, end)):
                temp_original_index.append(j)
        for j in range(remaining_tokens[i].__len__()):
            if(original_tokens_indexes[i][j] in temp_original_index):
                temp_answer_index.append(j)
        try:
            answer_indexes.append(temp_answer_index) # (temp_prep_ans_idx_range[0], temp_prep_ans_idx_range[-1] + 1), (begin, end)))
        except IndexError:
            answer_indexes.append([-16])

    """
    prepare x, y to train and test
    """
    from keras.utils import to_categorical

    flatten_embedded_sentences = embedded_sentences
    flatten_document_ids = np.hstack(document_ids)

    document_classes = []
    for doc_id in flatten_document_ids:
        document_classes.append(doc_id2class[doc_id])
    
    x = flatten_embedded_sentences
    y = to_categorical(document_classes)

    print(selected_tokenized_questions)

    # vectorize questions
    embedded_questions = []
    for i in range(selected_tokenized_questions.__len__()):
        temp_embedded_questions = []
        for j in range(selected_tokenized_questions[i].__len__()):
            try:
                print(selected_tokenized_questions[i])
                embedded_token = word_vectors[selected_tokenized_questions[i][j]]
                temp_embedded_questions.append(embedded_token)
            except:
                temp_embedded_questions.append(np.zeros(embedding_shape))
        while(temp_embedded_questions.__len__() < words_per_sentence):
            temp_embedded_questions.insert(0, np.zeros(embedding_shape))
        while(temp_embedded_questions.__len__() > words_per_sentence):
            temp_embedded_questions.pop()
        embedded_questions.append(np.asarray(temp_embedded_questions))

    x_questions = np.asarray(embedded_questions)
    print(x_questions)

    from keras.layers import Activation, Bidirectional, Dense, Input, LSTM, SpatialDropout1D, BatchNormalization
    from keras.models import load_model, Model
    from keras.optimizers import Adam

    lstm_output_size = 128

    vectorize_model = load_model('./models/' + str(words_per_sentence) + 'w-' + str(overlapping_words) + '-overlap-sentence-vectorization-model-' + str(512) + '.h5')
    vectorize_model.summary()

    # preprocess ของวินเนอร์ไม่เปลี่ยนตำแหน่งตัวอักษร
    # หาคำที่เป็นคำตอบก่อนจะได้ ['รา'] -> [468:469], ['บัต '] -> [470:473] แล้วค่อย remove stop words
    # {
    #   "question_id":3994,
    #   "question":"ปัตตานี เป็นจังหวัดในภาคใดของประเทศไทย",
    #   "answer":"ใต้","answer_begin_position ":125,
    #   "answer_end_position":128,
    #   "article_id":6865
    # }

    dense1_layer = Model(input=vectorize_model.input, output=vectorize_model.get_layer(index=4).output)
    dense1_layer_question_outputs = dense1_layer.predict(x_questions)

    print(dense1_layer_question_outputs.__len__())

    # vectorize all sentence in x
    dense1_layer_candidate_outputs = []
    for i in range(x.__len__()):
        temp = dense1_layer.predict(x[i])
        dense1_layer_candidate_outputs.append(temp)
    
    print(dense1_layer_candidate_outputs.__len__())

    distance_matrix = []
    for i in range(dense1_layer_candidate_outputs.__len__()):
        temp = []
        for j in range(dense1_layer_candidate_outputs[i].__len__()):
            distance = np.linalg.norm(dense1_layer_candidate_outputs[i][j] - dense1_layer_question_outputs[i])
            temp.append(distance)
        distance_matrix.append(temp)
    
    # print(distance_matrix)

    min_distance_indexes = []
    for i in range(distance_matrix.__len__()):
        min_index = np.asarray(distance_matrix[i]).argsort()[:distance_matrix[i].__len__()]
        min_distance_indexes.append(min_index)
    min_distance_indexes = np.asarray(min_distance_indexes)

    min_distances = []
    for i in range(min_distance_indexes.__len__()):
        temp_distance = []
        for idx in min_distance_indexes[i]:
            temp_distance.append(distance_matrix[i][idx])
        min_distances.append(np.asarray(temp_distance))
    min_distances = np.array(min_distances)

    # print(min_distance_indexes)

    # closest sentence range to original character position **********
    original_sentence_character_positions = []
    for i in range(min_distance_indexes.__len__()):
        temp_all_sentences = []
        for j in range(min_distance_indexes[i].__len__()):
            temp = []
            min_dist_idx = min_distance_indexes[i][j]
            sentence_range = m_words_sentence_ranges[i][min_dist_idx]
            for k in range(sentence_range[0], sentence_range[1]):
                character_position = original_tokens_ranges[i][k]
                # print(character_position)
                temp.append(character_position)
            temp_all_sentences.append(temp)
        original_sentence_character_positions.append(temp_all_sentences)
    
    # print(original_sentence_character_positions)

    candidates = []
    for i in range(original_sentence_character_positions.__len__()):
        answer_details = {}
        for j in range(original_sentence_character_positions[i].__len__()):
            begin_of_sentence = original_sentence_character_positions[i][j][0] # ***** answer for winner
            end_of_sentence = original_sentence_character_positions[i][j][-1] # ***** answer for winner
            print('question:', selected_plain_text_questions[i])
            print('rank['+ str(j) + '] answer sentence from program:', selected_plain_text_article[i][begin_of_sentence:end_of_sentence])
            print('begin:', begin_of_sentence, 'end:', end_of_sentence)
            answer_details = {
                "rank": i, 
                "question_id": selected_questions_numbers[i], 
                "question": selected_plain_text_questions[i],  
                "answer_begin_position ": original_sentence_character_positions[i][j][0], 
                "answer_end_position": original_sentence_character_positions[i][j][-1], 
                "article_id": selected_article_ids[i], 
                "similarity_score": float(min_distances[i][j])
            }
        candidates.append(answer_details)
    print(candidates)

    with open('./result/candidate_sentences.json', 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(candidates, f, ensure_ascii=False)

    # print(original_tokens_indexes)

    answer_masks = []
    answer_sentences = []
    answer_sentence_ranks = []
    for i in range(min_distance_indexes.__len__()):
        print('question:', i)
        answer_mask = []
        answer_sentence = []
        answer_sentence_rank = []
        for j in range(min_distance_indexes[i].__len__()):
            jth_closest_sentence = min_distance_indexes[i][j]
            temp_candidate_idx_range = m_words_sentence_ranges[i][jth_closest_sentence]
            candidate_idx_range = list(range(temp_candidate_idx_range[0], temp_candidate_idx_range[-1]))
            ans_idx_range = list(range(answer_indexes[i][0], answer_indexes[i][-1] + 1))
            has_answer = 0
            for k in range(ans_idx_range.__len__()):
                print('answer indexes:', ans_idx_range, 'sentence indexes:', candidate_idx_range)
                if(ans_idx_range[k] in candidate_idx_range):
                    has_answer = 1
                else:
                    has_answer = 0
                    break
                print(str(k) + '-th answer word is in sentence:', bool(has_answer))
            if(has_answer):
                print('rank:', jth_closest_sentence)
                answer_sentence.append([m_words_preprocessed_articles[i][jth_closest_sentence], candidate_idx_range])
            answer_mask.append(has_answer)
            answer_sentence_rank.append(jth_closest_sentence)
            answer_sentences.append(answer_sentence)
        answer_sentence_ranks.append(answer_sentence_rank)
        answer_masks.append(answer_mask)

    print(np.asarray(answer_sentence_ranks))
    print(np.asarray(answer_sentences))
    
    for i in range(answer_sentences.__len__()):
        for j in range(answer_sentences[i].__len__()):
            print('question:', selected_tokenized_questions[i])
            print('true answer: ', answer_sentences[i][j][0])
            print('rank:', answer_sentence_ranks[i][j])

    # # answer_sentence_charater_locations = []
    # # for i in range(answer_sentences.__len__()):
    #     # print(answer_sentences[i])
    
    # print(selected_tokenized_articles[10])
    # print(selected_tokenized_articles[10].__len__())
    # print(answer_sentences[10])

    # answer_sentence_character_positions = []
    # temp_answer_sentence_character_positions = []
    # for i in range(answer_sentences.__len__()):
    #     temp = []
    #     for j in range(answer_sentences[i].__len__()):
    #         temp.append((answer_sentences[i][j][1][0], answer_sentences[i][0][1][-1]))
    #     temp_answer_sentence_character_positions.append(temp)
    
    # for i in range(temp_answer_sentence_character_positions.__len__()):
    #     try:
    #         if(temp_answer_sentence_character_positions[i]):
    #             for j in range(temp_answer_sentence_character_positions[i].__len__()):
    #                 start = temp_answer_sentence_character_positions[i][j][0]
    #                 end = temp_answer_sentence_character_positions[i][j][-1]
    #                 answer_sentence_character_positions.append((original_preprocessed_cumulative_word_lengths[i][start], original_preprocessed_cumulative_word_lengths[i][end]))
    #         else:
    #             answer_sentence_character_positions.append((-18, -18))
    #     except IndexError:
    #         print(start, end, i)
    # print(answer_sentence_character_positions)
    # exit()

    # # answer_sentence_char_locs = []
    # # for i in range(answer_sentences.__len__()):
    # #     temp_char_loc = []
    # #     for j in range(answer_sentences[i].__len__()):
    # #         start_of_sentence_idx = answer_sentences[i][j][1][0]
    # #         print([id2word[idx] for idx in answer_sentences[i][j][0]], preprocessed_articles_sentences_index[i][start_of_sentence_idx])
    # #         print(preprocessed_cumulative_word_lengths[i].__len__())
    # #         end_of_sentence_idx = answer_sentences[i][j][1][-1]
    # #         print(end_of_sentence_idx)
    # #         temp_char_loc.append((preprocessed_cumulative_word_lengths[i][start_of_sentence_idx], preprocessed_cumulative_word_lengths[i][end_of_sentence_idx]))
    # #     answer_sentence_char_locs.append(temp_char_loc)
    # # print(answer_sentence_char_locs)
    
    # temp_answer_sentences = answer_sentences.copy()
    # for i in range(answer_sentences.__len__()):
    #     for j in range(answer_sentences[i].__len__()):
    #         temp_answer_sentences[i][j] = [id2word[idx] for idx in answer_sentences[i][j]]
    #         print(temp_answer_sentences[i][j])

    # ranks = []
    # for i in range(answer_masks.__len__()):
    #     temp_ranks = []
    #     for j in range(answer_masks[i].__len__()):
    #         if(answer_masks[i][j]):
    #             temp_ranks.append(j)
    #     if(not(temp_ranks)):
    #         temp_ranks.append(-18)
    #     ranks.append(temp_ranks)
    
    # flattened_ranks = [rank for ans in ranks for rank in ans]
    # # print(flattened_ranks)
    # ranks_ocurrences = dict(collections.Counter(flattened_ranks))
    # print(ranks_ocurrences)

    # proportion_ranks = []
    # for i in range(ranks.__len__()):
    #     proportion_rank = []
    #     for j in range(ranks[i].__len__()):
    #         print(ranks[i][j])
    #         proportion_rank.append(ranks[i][j] / padded_content_id_representation[i].__len__())
    #     proportion_ranks.append(proportion_rank)
    # print(proportion_ranks)
    # flattened_proportion_ranks = [rank for ans in proportion_ranks for rank in ans]

    # # for i in range(selected_questions.__len__()):
    #     # print(selected_questions[i])
    #     # print(answer_indexes[i][0])
    #     # for j in range(answer_sentences[i].__len__()):
    #         # answer_sentences[i][j] = [id2word[idx] for idx in answer_sentences[i][j]]
    #         # print(answer_sentences[i][j])
    #         # print('***')

    #         # print(list(range(candidate_idx_range[0], candidate_idx_range[-1])))
    #         # if()
    #     # print(i, '***')
    #         # if(preprocessed_articles_sentences_index[jth_closest_sentence]):
    # # print(min_distance_indexes.__len__())
    # # print(preprocessed_articles_sentences_index.__len__())
    # # print(answer_indexes)
    # # 
    # #     if(answer_matrix[i] != 0):
    # #     print(min_distance_indexes[i][answer_matrix[i]], min_distance_indexes[i].__len__())
    # #     ranks.append(min_distance_indexes[i][answer_matrix[i]])
    # #     proportion_ranks.append((min_distance_indexes[i][answer_matrix[i]]) / min_distance_indexes[i].__len__())
    # #     # else:

    # # print(proportion_ranks)

    # # print(np.asarray(ranks))
    # # # print(np.asarray(proportion_ranks))
    
    # fig1, axs1 = plt.subplots(1, tight_layout=True)
    # n, bins, patches = axs1.hist(flattened_ranks, bins='auto')
    # fracs = n / n.max()
    # norm = colors.Normalize(fracs.min(), fracs.max())
    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)
    # title = str(words_per_sentence) + '-Words Sentence Model (from: ' + str(n_samples) + ' Samples)'
    # fig1.suptitle('N-ranks similarity between question and sentences in article', fontsize=12, fontweight='bold')
    # axs1.set_title(title)
    # axs1.set_xlabel('Closest to sentence rank (n-th)')
    # axs1.set_ylabel('Occurrence')
    # plt.savefig('./tmp1-' + str(words_per_sentence) + '.png')
    # plt.show()

    # fig2, axs2 = plt.subplots(1)
    # n, bins, patches = axs2.hist(flattened_proportion_ranks, bins='auto')
    # fracs = n / n.max()
    # norm = colors.Normalize(fracs.min(), fracs.max())
    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)
    # axs2.set_title(title)
    # fig2.suptitle('N-ranks similarity between question and sentences in article', fontsize=12, fontweight='bold')
    # axs2.set_xlabel('Closest to sentence rank (n-th) / Article length (sentences)')
    # axs2.set_ylabel('Occurrence')
    # plt.savefig('./tmp2-' + str(words_per_sentence) + '.png')
    # plt.show()
