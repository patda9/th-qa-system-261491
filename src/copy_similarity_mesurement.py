import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import preprocessing as prep
import re
import sentence_vectorization as sv
import os

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

np.random.seed(0)

if(__name__ == '__main__'):
    dataset_path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/'
    text_path = 'C:/Users/Patdanai/Desktop/documents-nsc/'
    dataset = os.listdir(dataset_path)
    n_samples = 512

    models_ws_archs_path = 'C:/Users/Patdanai/Desktop/492/th-qa-system-261491/models'
    model_files = os.listdir(models_ws_archs_path)

    questions = []
    with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
        questions = json.load(f)

    with open('./../new_sample_questions_answer.json', 'r', encoding='utf-8', errors='ignore') as f:
        answer_doc_id = json.load(f)

    answer_details = {}
    with open('./../new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        answer_details = json.load(f)

    last_doc_id = 282972
    
    # get first 512 samples from questions
    count = 0
    selected_article_ids = []
    selected_questions_ids = []
    selected_questions_plain_text = []
    for q in answer_details['data']:
        if(q['article_id'] < last_doc_id and count < 512):
            selected_article_ids.append(q['article_id'])
            selected_questions_ids.append(q['question_id'])
            selected_questions_plain_text.append(q['question'])
            count += 1
    
    # print(list(zip(selected_questions_ids, selected_article_ids)))

    # load each article content
    selected_articles = [] # plain text
    selected_tokenized_articles = [] # tokens
    for i in selected_article_ids:
        text_dataset_path = os.path.join(dataset_path, str(i) + '.json')
        with open(text_dataset_path, 'r', encoding='utf-8') as f:
            article = json.load(f)        
        selected_articles.append(''.join(article))
        selected_tokenized_articles.append(article)

    # remove noise from questions
    selected_questions = []
    for i in range(questions.__len__()):
        if((i+1) in selected_questions_ids): # i+1: first question starts at 1 
            questions[i] = [w for w in questions[i] if not(w is ' ')]
            selected_questions.append(questions[i])

    # print(selected_questions[-1])

    preprocessed_articles = []
    j = 1
    for i in selected_article_ids:
        file_path = os.path.join(dataset_path, str(i) + '.json')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            data = prep.remove_noise(data)
        preprocessed_articles.append(data)
        j += 1
    
    # preprocessed_articles[i][0] => (index of /*(starts from 0)*/, word)
    # preprocessed_articles[i][1] => list of cumulative string length (end position of words)

    remaining_words = []
    cumulative_word_lengths = []
    original_cumulative_word_lengths = []
    for i in range(preprocessed_articles.__len__()):
        remaining_words.append(preprocessed_articles[i][0])
        cumulative_word_lengths.append(preprocessed_articles[i][1])
        original_cumulative_word_lengths.append(preprocessed_articles[i][2])
    
    print(remaining_words[0])
    print(original_cumulative_word_lengths[0][151])

    original_preprocessed_cumulative_word_lengths = []
    for i in range(remaining_words.__len__()):
        temp = []
        for j in range(remaining_words[i].__len__()):
            temp.append(original_cumulative_word_lengths[i][remaining_words[i][j][0]])
        original_preprocessed_cumulative_word_lengths.append(temp)

    print(original_preprocessed_cumulative_word_lengths[151])

    # print(remaining_words[0])
    # print(cumulative_word_lengths[1])

    vocabularies = [article[i][1] for article in remaining_words for i in range(article.__len__())]
    
    word2id = {}
    word2id['<PAD>'] = 0
    for (i, w) in enumerate(set(vocabularies), 1):
        try:
            word2id[w] = i
        except ValueError:
            pass

    id2word = {idx: w for w, idx in word2id.items()}
    
    # print(word2id)

    # preprocess questions
    pattern = re.compile(r"[^\u0E00-\u0E7F^0-9^ \t^.^,]")
    selected_question_id_representation = []
    for i in range(selected_questions.__len__()):
        temp_list = []
        for j in range(selected_questions[i].__len__()):
            char_to_remove = re.findall(pattern, selected_questions[i][j])
            temp_word = ''
            for k in range(selected_questions[i][j].__len__()):
                if(not(selected_questions[i][j] in char_to_remove)):
                    temp_word += selected_questions[i][j][k]
            temp_list.append(temp_word)
        selected_questions[i] = temp_list

    for q in selected_questions:
        temp = []
        for w in q:
            try:
                temp.append(word2id[w])
            except KeyError:
                pass
        selected_question_id_representation.append(temp)

    # print(selected_question_id_representation)
    # print(selected_articles[-1])

    content_id_representation = []
    for q in remaining_words:
        temp = []
        for w in q:
            try:
                temp.append(word2id[w[1]])
            except KeyError:
                pass
        content_id_representation.append(temp)

    words_per_sentence = 20
    overlap_flag = False
    overlapping_words = words_per_sentence // 2

    temp_preprocessed_article = sv.k_words_separate(words_per_sentence, content_id_representation, overlap=overlap_flag)
    preprocessed_article_id_representation = temp_preprocessed_article[0].copy()
    preprocessed_articles_sentences_index = temp_preprocessed_article[1].copy()

    # print(preprocessed_article_id_representation[0])
    # print(preprocessed_articles_sentences_index[0])

    from keras.preprocessing import sequence
    
    padded_selected_question_id_representation = sequence.pad_sequences(selected_question_id_representation, maxlen=words_per_sentence)
    for i in range(preprocessed_article_id_representation.__len__()):
        preprocessed_article_id_representation[i] = sequence.pad_sequences(preprocessed_article_id_representation[i], maxlen=words_per_sentence)

    padded_content_id_representation = preprocessed_article_id_representation.copy()
    
    # print(padded_selected_question_id_representation[-1])
    # print(preprocessed_article_id_representation)

    from keras.layers import Activation, Bidirectional, Dense, Embedding, Flatten, InputLayer, LSTM, TimeDistributed
    from keras.layers.core import Masking
    from keras.models import load_model, Model, Sequential
    from keras.optimizers import Adam
    from keras.utils import to_categorical

    if(overlap_flag):
        model = load_model('./models/' + str(words_per_sentence) + 'w-' + str(overlapping_words) + '-overlap-sentence-vectorization-model-' + str(n_samples) + '.h5')
    else:
        model = load_model('./models/' + str(words_per_sentence) + 'w-sentence-vectorization-model-' + str(n_samples) + '.h5')

    vectorize_model = Sequential()
    vectorize_model.add(Embedding(word2id.__len__(), 64, input_length=words_per_sentence, name='e'))
    vectorize_model.add(Masking(mask_value=0, input_shape=(words_per_sentence, 1)))
    vectorize_model.add(Bidirectional(LSTM(128)))
    vectorize_model.add(Dense(64, activation='relu'))
    vectorize_model.layers[2].set_weights(model.layers[2].get_weights())
    vectorize_model.layers[3].set_weights(model.layers[3].get_weights())
    vectorize_model.summary()

    answer_char_locs = []
    for i in range(selected_questions_ids.__len__()):
        for q in answer_details['data']:
            if(selected_questions_ids[i] == q['question_id']):
                answer_begin_pos = q['answer_begin_position '] - 1 # pos - 1 to refer array index
                answer_end_pos = q['answer_end_position'] - 1
                answer_char_locs.append((answer_begin_pos, answer_end_pos))

    answers = []
    for i in range(n_samples):
        answer = selected_articles[i][answer_char_locs[i][0]:answer_char_locs[i][1]]
        answers.append(answer)

    # print(remaining_words[0], cumulative_word_lengths[0])

    # find ans index
    answer_indexes = []
    for i in range(remaining_words.__len__()):
        temp_indexes = []
        temp_answer_index = []
        for j in range(cumulative_word_lengths[i].__len__()):
            begin = answer_char_locs[i][0] # + 1 to turn back to original position
            end = answer_char_locs[i][1] + 1 # range = end -1
            eca = cumulative_word_lengths[i][j] # ending character of the answer
            if(eca in range(begin, end)):
                temp_indexes.append(j)
        temp_prep_ans_idx_range = []
        for j in range(remaining_words[i].__len__()):
            if(remaining_words[i][j][0] in temp_indexes):
                temp_answer_index.append(remaining_words[i][j])
                temp_prep_ans_idx_range.append(j)
        try:
            answer_indexes.append((temp_answer_index, (temp_prep_ans_idx_range[0], temp_prep_ans_idx_range[-1] + 1), (begin, end)))
        except IndexError:
            answer_indexes.append(([(-18, -18)], (-18, -18), (begin, end)))
    
    # print(answer_indexes) # [indexes of answer, (start, ending characters)]
    
    # print(preprocessed_article_id_representation[-1])

    # for i in range(padded_selected_question_id_representation.__len__()):
    #     selected_question_id_representation[i] = [id2word[idx] for idx in selected_question_id_representation[i]]
    #     print(selected_question_id_representation[i])

    # preprocess ของวินเนอร์ไม่เปลี่ยนตำแหน่งตัวอักษร
    # หาคำที่เป็นคำตอบก่อนจะได้ ['รา'] -> [468:469], ['บัต '] -> [470:473] แล้วค่อย remove stop words
    # {
    #   "question_id":3994,
    #   "question":"ปัตตานี เป็นจังหวัดในภาคใดของประเทศไทย",
    #   "answer":"ใต้","answer_begin_position ":125,
    #   "answer_end_position":128,
    #   "article_id":6865
    # }

    dense1_layer = Model(input=vectorize_model.input, output=vectorize_model.get_layer(index=3).output)
    dense1_layer_question_outputs = dense1_layer.predict(padded_selected_question_id_representation)

    # vectorize all sentence in 
    dense1_layer_answer_outputs = []
    for i in range(padded_content_id_representation.__len__()):
        temp = dense1_layer.predict(padded_content_id_representation[i])
        dense1_layer_answer_outputs.append(temp)
    
    dense1_layer_question_outputs = dense1_layer.predict(padded_selected_question_id_representation)
    
    # print(dense1_layer_answer_outputs.__len__())
    # print(dense1_layer_question_outputs.__len__())

    distance_matrix = []
    for i in range(dense1_layer_answer_outputs.__len__()):
        temp = []
        for j in range(dense1_layer_answer_outputs[i].__len__()):
            distance = np.linalg.norm(dense1_layer_answer_outputs[i][j] - dense1_layer_question_outputs[i])
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

    answer_masks = []
    answer_sentences = []
    answer_sentence_ranks = []
    for i in range(min_distance_indexes.__len__()):
        answer_mask = []
        answer_sentence = []
        answer_sentence_rank = []
        for j in range(min_distance_indexes[i].__len__()):
            jth_closest_sentence = min_distance_indexes[i][j]
            temp_candidate_idx_range = preprocessed_articles_sentences_index[i][jth_closest_sentence]
            candidate_idx_range = list(range(temp_candidate_idx_range[0], temp_candidate_idx_range[-1]))
            # print(candidate_idx_range)
            ans_idx_range = list(range(answer_indexes[i][1][0], answer_indexes[i][1][1]))
            has_answer = 0
            for k in range(ans_idx_range.__len__()):
                # print('answer indexes:', ans_idx_range, 'sentence indexes', candidate_idx_range)
                if(ans_idx_range[k] in candidate_idx_range):
                    has_answer = 1
                else:
                    has_answer = 0
                    break
                # print(str(k) + '-th answer word', 'is in sentence:', has_answer)
            if(has_answer):
                # print('rank:', jth_closest_sentence)
                answer_sentence.append((preprocessed_article_id_representation[i][jth_closest_sentence], candidate_idx_range))
            # print(answer_sentence)
            answer_mask.append(has_answer)
            answer_sentence_rank.append(jth_closest_sentence)
        answer_sentences.append(answer_sentence)
        answer_sentence_ranks.append(answer_sentence_rank)
        answer_masks.append(answer_mask)
    
    # for i in range(answer_sentences.__len__()):
    #     for j in range(answer_sentences[i].__len__()):
    #         print('question:', selected_questions[i])
    #         print('true answer: ', [id2word[idx] for idx in answer_sentences[i][j][0]])
    #         print('rank:', answer_sentence_ranks[i][j])

    # print(preprocessed_cumulative_word_lengths[0][0])

    # answer_sentence_charater_locations = []
    # for i in range(answer_sentences.__len__()):
        # print(answer_sentences[i])
    
    print(selected_tokenized_articles[10])
    print(answer_sentences[10])

    answer_sentence_character_positions = []
    temp_answer_sentence_character_positions = []
    for i in range(answer_sentences.__len__()):
        temp = []
        for j in range(answer_sentences[i].__len__()):
            temp.append((answer_sentences[i][j][1][0], answer_sentences[i][0][1][-1]))
        temp_answer_sentence_character_positions.append(temp)
    
    for i in range(temp_answer_sentence_character_positions.__len__()):
        try:
            if(temp_answer_sentence_character_positions[i]):
                for j in range(temp_answer_sentence_character_positions[i].__len__()):
                    start = temp_answer_sentence_character_positions[i][j][0]
                    end = temp_answer_sentence_character_positions[i][j][-1]
                    answer_sentence_character_positions.append((original_preprocessed_cumulative_word_lengths[i][start], original_preprocessed_cumulative_word_lengths[i][end]))
            else:
                answer_sentence_character_positions.append((-18, -18))
        except IndexError:
            print(start, end, remaining_words.__len__(), i)
    print(answer_sentence_character_positions)
    exit()

    # answer_sentence_char_locs = []
    # for i in range(answer_sentences.__len__()):
    #     temp_char_loc = []
    #     for j in range(answer_sentences[i].__len__()):
    #         start_of_sentence_idx = answer_sentences[i][j][1][0]
    #         print([id2word[idx] for idx in answer_sentences[i][j][0]], preprocessed_articles_sentences_index[i][start_of_sentence_idx])
    #         print(preprocessed_cumulative_word_lengths[i].__len__())
    #         end_of_sentence_idx = answer_sentences[i][j][1][-1]
    #         print(end_of_sentence_idx)
    #         temp_char_loc.append((preprocessed_cumulative_word_lengths[i][start_of_sentence_idx], preprocessed_cumulative_word_lengths[i][end_of_sentence_idx]))
    #     answer_sentence_char_locs.append(temp_char_loc)
    # print(answer_sentence_char_locs)
    
    temp_answer_sentences = answer_sentences.copy()
    for i in range(answer_sentences.__len__()):
        for j in range(answer_sentences[i].__len__()):
            temp_answer_sentences[i][j] = [id2word[idx] for idx in answer_sentences[i][j]]
            print(temp_answer_sentences[i][j])

    ranks = []
    for i in range(answer_masks.__len__()):
        temp_ranks = []
        for j in range(answer_masks[i].__len__()):
            if(answer_masks[i][j]):
                temp_ranks.append(j)
        if(not(temp_ranks)):
            temp_ranks.append(-18)
        ranks.append(temp_ranks)
    
    flattened_ranks = [rank for ans in ranks for rank in ans]
    # print(flattened_ranks)
    ranks_ocurrences = dict(collections.Counter(flattened_ranks))
    print(ranks_ocurrences)

    proportion_ranks = []
    for i in range(ranks.__len__()):
        proportion_rank = []
        for j in range(ranks[i].__len__()):
            print(ranks[i][j])
            proportion_rank.append(ranks[i][j] / padded_content_id_representation[i].__len__())
        proportion_ranks.append(proportion_rank)
    print(proportion_ranks)
    flattened_proportion_ranks = [rank for ans in proportion_ranks for rank in ans]

    # for i in range(selected_questions.__len__()):
        # print(selected_questions[i])
        # print(answer_indexes[i][0])
        # for j in range(answer_sentences[i].__len__()):
            # answer_sentences[i][j] = [id2word[idx] for idx in answer_sentences[i][j]]
            # print(answer_sentences[i][j])
            # print('***')

            # print(list(range(candidate_idx_range[0], candidate_idx_range[-1])))
            # if()
        # print(i, '***')
            # if(preprocessed_articles_sentences_index[jth_closest_sentence]):
    # print(min_distance_indexes.__len__())
    # print(preprocessed_articles_sentences_index.__len__())
    # print(answer_indexes)
    # 
    #     if(answer_matrix[i] != 0):
    #     print(min_distance_indexes[i][answer_matrix[i]], min_distance_indexes[i].__len__())
    #     ranks.append(min_distance_indexes[i][answer_matrix[i]])
    #     proportion_ranks.append((min_distance_indexes[i][answer_matrix[i]]) / min_distance_indexes[i].__len__())
    #     # else:

    # print(proportion_ranks)

    # print(np.asarray(ranks))
    # # print(np.asarray(proportion_ranks))
    
    fig1, axs1 = plt.subplots(1, tight_layout=True)
    n, bins, patches = axs1.hist(flattened_ranks, bins='auto')
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    title = str(words_per_sentence) + '-Words Sentence Model (from: ' + str(n_samples) + ' Samples)'
    fig1.suptitle('N-ranks similarity between question and sentences in article', fontsize=12, fontweight='bold')
    axs1.set_title(title)
    axs1.set_xlabel('Closest to sentence rank (n-th)')
    axs1.set_ylabel('Occurrence')
    plt.savefig('./tmp1-' + str(words_per_sentence) + '.png')
    plt.show()

    fig2, axs2 = plt.subplots(1)
    n, bins, patches = axs2.hist(flattened_proportion_ranks, bins='auto')
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    axs2.set_title(title)
    fig2.suptitle('N-ranks similarity between question and sentences in article', fontsize=12, fontweight='bold')
    axs2.set_xlabel('Closest to sentence rank (n-th) / Article length (sentences)')
    axs2.set_ylabel('Occurrence')
    plt.savefig('./tmp2-' + str(words_per_sentence) + '.png')
    plt.show()
