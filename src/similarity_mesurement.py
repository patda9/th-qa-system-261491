import json
import matplotlib.pyplot as plt
import numpy as np
import preprocessing as prep
import sentence_vectorization as sv
import os

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
    for q in answer_details['data']:
        if(q['article_id'] < last_doc_id and count < 512):
            selected_article_ids.append(q['article_id'])
            selected_questions_ids.append(q['question_id'])
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

    # print(selected_articles)
    # print(selected_tokenized_articles)

    # remove whitespaces from questions
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
            data = prep.remove_xml(data)
            data = prep.remove_stop_words(data)
        preprocessed_articles.append(data)
        j += 1
    
    # print(preprocessed_articles[1])

    vocabularies = [word for article in list(preprocessed_articles) for word in article]
    
    word2id = {}
    word2id['<PAD>'] = 0
    word2id['<NIV>'] = 1
    for (i, w) in enumerate(set(vocabularies), 2):
        try:
            word2id[w] = i
        except ValueError:
            print('niv')
            word2id[w] = 1

    id2word = {idx: w for w, idx in word2id.items()}
    
    # print(word2id)

    selected_question_id_representation = []
    for q in selected_questions:
        temp = []
        for w in q:
            try:
                temp.append(word2id[w])
            except KeyError:
                temp.append(word2id['<NIV>'])
        selected_question_id_representation.append(temp)

    # print(selected_question_id_representation)
    # print(selected_articles[-1])

    content_id_representation = []
    for q in preprocessed_articles:
        temp = []
        for w in q:
            try:
                temp.append(word2id[w])
            except KeyError:
                temp.append(word2id['<NIV>'])
        content_id_representation.append(temp)

    words_per_sentence = 20
    overlap_flag = False
    overlapping_words = words_per_sentence // 2

    preprocessed_article_id_representation = sv.k_words_separate(words_per_sentence, content_id_representation, overlap=overlap_flag)

    from keras.preprocessing import sequence
    
    padded_selected_question_id_representation = sequence.pad_sequences(selected_question_id_representation, maxlen=words_per_sentence)
    for i in range(preprocessed_article_id_representation.__len__()):
        preprocessed_article_id_representation[i] = sequence.pad_sequences(preprocessed_article_id_representation[i], maxlen=words_per_sentence)

    padded_content_id_representation = preprocessed_article_id_representation.copy()
    
    # print(padded_selected_question_id_representation[-1])
    print(preprocessed_article_id_representation)

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
                answer_end_pos = q['answer_end_position']
                answer_char_locs.append(range(answer_begin_pos, answer_end_pos))

    answers = []
    for i in range(n_samples):
        answer = selected_articles[i][answer_char_locs[i][0]:answer_char_locs[i][-1]]
        answers.append(answer)

    
    # find ans index
    answer_indexes = []
    for i in range(selected_tokenized_articles.__len__()):
        current_char_loc = 0
        for j in range(selected_tokenized_articles[i].__len__()):
            for c in selected_tokenized_articles[i][j]:
                if(current_char_loc == answer_char_locs[i][0]):
                    answer_indexes.append([j, answers[i]])
                current_char_loc += 1

    # print(answer_indexes)

    answers_after_removed_stop_words = []
    for i in range(preprocessed_articles.__len__()):
        answer_index_after_removed_stop_words = 0
        all_answer_indexes = []
        for j in range(preprocessed_articles[i].__len__()):
            if(preprocessed_articles[i][j].strip() == preprocessed_articles[i][1].strip()):
                all_answer_indexes.append(j)
        answer_index_after_removed_stop_words = 0
        for k in all_answer_indexes:
            if(np.abs(answer_indexes[i][0] - k) < np.abs(answer_indexes[i][0] - answer_index_after_removed_stop_words)):
                answer_index_after_removed_stop_words = k
        answers_after_removed_stop_words.append(answer_index_after_removed_stop_words)
    
    # print(answers_after_removed_stop_words)

    # print(preprocessed_article_id_representation[-1])

    for i in range(padded_selected_question_id_representation.__len__()):
        selected_question_id_representation[i] = [id2word[idx] for idx in selected_question_id_representation[i]]
        # print(selected_question_id_representation[i])

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
    
    # print(dense1_layer_answer_outputs)

    distance_matrix = []
    for i in range(padded_content_id_representation.__len__()):
        temp = []
        for j in range(padded_content_id_representation[i].__len__()):
            distance = np.linalg.norm(padded_content_id_representation[i][j] - padded_selected_question_id_representation[i])
            temp.append(distance)
        distance_matrix.append(temp)

    min_distance_indexes = []
    for i in range(distance_matrix.__len__()):
        min_index = np.asarray(distance_matrix[i]).argsort()[:distance_matrix[i].__len__()]
        min_distance_indexes.append(min_index)

    min_distance_indexes = np.asarray(min_distance_indexes)
    
    # print(dense1_layer_question_outputs.__len__())
    # print(dense1_layer_answer_outputs[0].__len__())

    print(min_distance_indexes)
    # print(distance_matrix)
    # print(np.asarray(answers_after_removed_stop_words) // words_per_sentence)

    answer_matrix = np.asarray(answers_after_removed_stop_words) // words_per_sentence
    
    print(answer_matrix)

    ranks = []
    proportion_ranks = []
    for i in range(min_distance_indexes.__len__()):
        print(min_distance_indexes[i][answer_matrix[i]])
        # for j in range(min_distance_indexes[i].__len__()):
        ranks.append(min_distance_indexes[i][answer_matrix[i]] + 1)
        proportion_ranks.append(float((min_distance_indexes[i][answer_matrix[i]] + 1) / min_distance_indexes[i].__len__()))

    # print(np.asarray(ranks))
    # print(np.asarray(proportion_ranks))
    
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(ranks, bins='auto')
    axs[1].hist(proportion_ranks, bins='auto')
    plt.savefig('./tmp2.png')