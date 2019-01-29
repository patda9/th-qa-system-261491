import json
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
    print(model_files)

    questions = []
    with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
        questions = json.load(f)

    with open('./../new_sample_questions_answer.json', 'r', encoding='utf-8', errors='ignore') as f:
        answer_doc_id = json.load(f)

    answer_details = {}
    with open('./../new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        answer_details = json.load(f)

    last_doc_id = 282972

    sample_articles_id = []

    for q in answer_details['data']:
        if(q['article_id'] < last_doc_id):
            sample_articles_id.append(q['article_id'])

    sample_question_ans = []
    selected_questions_no = []
    for i in range(n_samples):
        randomed_question = np.random.randint(questions.__len__())
        while(answer_doc_id[randomed_question] > last_doc_id or randomed_question in selected_questions_no or not(answer_doc_id[randomed_question] in sample_articles_id)): # limited preprocessed corpus
            randomed_question = np.random.randint(questions.__len__())
        sample_question_ans.append((randomed_question + 1, answer_doc_id[randomed_question])) # question_ids start from 0 (+1)
        selected_questions_no.append(randomed_question)
    sample_question_ans = sorted(sample_question_ans)
    print(selected_questions_no)

    answer_char_locs = []

    article_samples = []
    tokenized_article_samples = []
    for t in sample_question_ans:
        text_dataset_path = os.path.join(dataset_path, str(t[1]) + '.json')
        with open(text_dataset_path, 'r', encoding='utf-8') as f:
            article = json.load(f)
        tokenized_article_samples.append(article)
        article_samples.append(''.join(article))

    #####
    # tokenized_article_samples = sv.k_words_separate(30, tokenized_article_samples)
    # print(tokenized_article_samples)

    for i in range(questions.__len__()):
        q = [question for question in questions[i] if(question.strip())]

    ## load input question sentence and output answer sentence
    for i in range(questions.__len__()):
        questions[i] = [w for w in questions[i] if not(w is ' ')]
    
    selected_questions = []
    for i in range(sample_question_ans.__len__()):
        for q in answer_details['data']:
            if(selected_questions_no[i] == q['question_id']):
                answer_begin_pos = q['answer_begin_position '] - 1 # pos - 1 to refer array index
                answer_end_pos = q['answer_end_position']
                answer_char_locs.append(range(answer_begin_pos, answer_end_pos))
                selected_questions.append(questions[i])

    # get article ids from directory path
    print(selected_questions)
    exit()

    ## get words/vocabularies from file
    articles = []
    i = 1
    for t in sample_question_ans:
        file_path = os.path.join(dataset_path, str(t[1]) + '.json')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            data = prep.remove_xml(data)
            data = prep.remove_stop_words(data)
        articles.append(data)
        i += 1

    vocabularies = [word for article in list(articles) for word in article]

    ### these commented blocks require preprocessed vocabs from corpus in .json
    # with open('./vocabs.json', 'w', encoding='utf-8', errors='ignore') as o:
    #     json.dump(vocabs_out, o)

    # with open('./vocabs.json', 'r', encoding='utf-8', errors='ignore') as f:
    #     vocabs = json.load(f).values()

    # print(articles)
    # print(articles.__len__())

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

    id_representation = []
    for article in articles:
        try:
            id_representation.append([word2id[w] for w in article])
        except KeyError:
            id_representation.append([word2id['<NIV>'] for w in article])

    question_id_representation = []
    p = 0
    for q in selected_questions:
        try:
            question_id_representation.append([word2id[w] for w in q])
            p += 1
        except KeyError:
            print('niv in question')
            print(q, selected_questions_no[p])
            question_id_representation.append([word2id['<NIV>'] for w in q])
            p += 1
    # print(question_id_representation)
    exit()
    
    words_per_sentence = 20
    overlap_flag = True
    overlapping_words = words_per_sentence // 2

    # id_representation = sv.k_words_separate(words_per_sentence, id_representation, overlap=overlap_flag)
    # print(question_id_representation)

    from keras.preprocessing import sequence
    
    padded_question_id_representation = sequence.pad_sequences(question_id_representation, maxlen=words_per_sentence)

    from keras.layers import Activation, Bidirectional, Dense, Embedding, Flatten, InputLayer, LSTM, TimeDistributed
    from keras.layers.core import Masking
    from keras.models import load_model, Model, Sequential
    from keras.optimizers import Adam
    from keras.utils import to_categorical

    model = None
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

    dense1_layer = Model(input=vectorize_model.input, output=vectorize_model.get_layer(index=3).output)
    dense1_layer_output = dense1_layer.predict(padded_question_id_representation)
    print(dense1_layer_output.__len__())

    answers = []
    for i in range(n_samples):
        answer = article_samples[i][answer_char_locs[i][0]:answer_char_locs[i][-1]]
        # print('question_id:', sample_question_ans[i][0], 'answer:', answer)
        answers.append(answer)

    answer_indexes = []
    ## find ans index
    for i in range(tokenized_article_samples.__len__()):
        current_char_loc = 0
        for j in range(tokenized_article_samples[i].__len__()):
            for c in tokenized_article_samples[i][j]:
                if(current_char_loc == answer_char_locs[i][0]):
                    answer_indexes.append([j, answers[i]])
                    print('index:', j, answer_char_locs[i][0], c, answers[i])
                current_char_loc += 1

    # print(answer_indexes)
    answers_after_removed_stop_words = []
    for i in range(articles.__len__()):
        answer_index_after_removed_stop_words = 0
        all_answer_indexes = []
        print(answer_indexes[i][1], sample_question_ans[i][1])
        for j in range(articles[i].__len__()):
            if(articles[i][j].strip() == answer_indexes[i][1].strip()):
                print(articles[i][j], answer_indexes[i][1])
                all_answer_indexes.append(j)
        # print('*', answer_indexes[i][0], all_answer_indexes)
        answer_index_after_removed_stop_words = 0
        for k in all_answer_indexes:
            if(np.abs(answer_indexes[i][0] - k) < np.abs(answer_indexes[i][0] - answer_index_after_removed_stop_words)):
                # print(np.abs(answer_indexes[i][0] - k))
                answer_index_after_removed_stop_words = k
        answers_after_removed_stop_words.append(answer_index_after_removed_stop_words)
    print(answers_after_removed_stop_words)

    id_representation = sv.k_words_separate(words_per_sentence, id_representation, overlap=overlap_flag)
    for i in range(id_representation.__len__()):
        print(padded_question_id_representation[i])
        # print(id_representation[i])
    for i in range(question_id_representation.__len__()):
        question_id_representation[i] = [id2word[idx] for idx in question_id_representation[i]]
        print(question_id_representation[i])

    # preprocess ของวินเนอร์ไม่เปลี่ยนตำแหน่งตัวอักษร
    # หาคำที่เป็นคำตอบก่อนจะได้ ['รา'] -> [468:469], ['บัต '] -> [470:473] แล้วค่อย remove stop words
    # {
    #   "question_id":3994,
    #   "question":"ปัตตานี เป็นจังหวัดในภาคใดของประเทศไทย",
    #   "answer":"ใต้","answer_begin_position ":125,
    #   "answer_end_position":128,
    #   "article_id":6865
    # }

