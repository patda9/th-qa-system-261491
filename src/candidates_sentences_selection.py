import json
import numpy as np
import os
import preprocessing as prep
import sentence_vectorization as sv

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

def load_json_doc_ids(path='C:/Users/Patdanai/Workspace/th-qa-system-261491/data/'):
    with open(path, 'r') as f:
        candidate_ids = json.load(f)
    
    return candidate_ids
    
def load_corpus_word_vectors(path='C:/Users/Patdanai/Downloads/th-qasys-db/word_vectors_model/word2vec.model'):
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
    
    return questions

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
    DOCUMENTS_PATH = 'C:/Users/Patdanai/Downloads/th-qasys-db/tokenized_wiki_corpus/'
    SV_MODEL_PATH = 'C:/Users/Patdanai/Downloads/th-qasys-db/sentence_vectorization_models/20w-10-overlap-sentence-vectorization-model-768-16.h5'
    WV_PATH = 'D:/Users/Patdanai/th-qasys-db/preprocessed_corpus_wv/'
    WV_MODEL_PATH = 'C:/Users/Patdanai/Downloads/th-qasys-db/word_vectors_model/word2vec.model'
    
    question_path = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/data/ThaiQACorpus-EvaluationDataset-tokenize.json'
    
    WORDS_PER_SENTENCE = 20
    OVERLAPPING_WORDS = WORDS_PER_SENTENCE // 2

    

    word_vectors = load_corpus_word_vectors(path=WV_MODEL_PATH)
    sv_model = load_sentence_vectorization_model(SV_MODEL_PATH)
    sv_layer = get_sentence_vectorization_layer(sv_model)

    candidate_document_ids = [[473065, 289367]]
    tokenized_questions = load_tokenized_questions(question_path)

    # implement small batch processing (1 question/batch)
    for i in range(candidate_document_ids.__len__()): # question
        # print('Processing question [' + str(i) + '/' + str(candidate_document_ids.__len__()) + '] candidate documents. \r', end='')
        
        documents_index = [] # original one
        documents_legths = [] # original one
        array_of_wvs = []
        for j in range(candidate_document_ids[i].__len__()): # candidate doc
            original_index, original_lengths = get_original_token_positions(candidate_document_ids[i][j], DOCUMENTS_PATH)
            array_of_wvs.append(load_document_word_vectors(candidate_document_ids[i][j], WV_PATH))
            documents_index.append(original_index)
            documents_legths.append(original_lengths)
        # print(len(array_of_wvs))
        # print(len(documents_index))
        # print(len(documents_legths))

        # make sample becomes batch (size 1) for feeding through sv_layer
        # embedded_question = np.expand_dims(vectorize_question_tokens(tokenized_questions[i], word_vectors), axis=0) # question_tokens => [question_tokens]
        # question_vector = sv_layer.predict(embedded_question).flatten()
        # print(embedded_question.shape)
        # print(question_vector.shape)

        m_tokens_groupping = sv.m_words_separate(WORDS_PER_SENTENCE, array_of_wvs, overlapping_words=OVERLAPPING_WORDS, question_number=i)
        m_tokens_documents = m_tokens_groupping[0]
        m_tokens_ranges = np.asarray(m_tokens_groupping[1])
        
        print(m_tokens_documents)
        for i in m_tokens_documents:
            print('i', i.__len__())
            for j in i:
                print('j', j.__len__())
        
        # print(m_tokens_ranges.__len__())

        candidate_sentence_vectors = []
        for i in range(len(m_tokens_documents)):
            candidate_sentence_vectors.append(sv_layer.predict(m_tokens_documents[i]))
        print(len(candidate_sentence_vectors))

    exit()

    ### sentence, question vectorization


wv_model = Word2Vec.load('C:/Users/Patdanai/Desktop/492/word2vec.model')
word_vectors = wv_model.wv

MAX_NUMBER_OF_WORDS = word_vectors.vocab.__len__()
MAX_SEQUENCE_LENGHT = WORDS_PER_SENTENCE

EMBEDDING_SHAPE = word_vectors['มกราคม'].shape # use as word vector's dimension

# questions vectorization goes here
questions = []
# change path to tokenized questions

# with open('./ThaiQACorpus-EvaluationDataset-tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
#     questions = json.load(f)

with open('././../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
    questions = json.load(f)

tokenized_questions = []
for i in range(questions.__len__()):
    questions[i] = [w for w in questions[i] if not(w is ' ')]
    tokenized_questions.append(questions[i])

print('Vectorizing input questions 0.')
embedded_questions = cl.vectorize_questions(tokenized_questions, word_vectors, embedding_shape=EMBEDDING_SHAPE, words_per_sentence=WORDS_PER_SENTENCE)
embedded_questions = np.asarray(embedded_questions)
print('Vectorizing input questions 1.')
question_vectors = dense_layer.predict(embedded_questions)

question_idx = 0
candidate_answers = []
candidate_character_positions = []
distance_scores = []
for question_candidates in m_words_preprocessed_documents[question_idx:]: # for each question
    print('question id:', question_idx)
    embedded_candidate = cl.vectorize_words(question_candidates, word_vectors, embedding_shape=EMBEDDING_SHAPE)
    try:
        character_positions, min_distance_matrix, sentence_indexes = cl.locate_candidates_sentence(question_vectors[question_idx], embedded_candidate, m_words_sentence_ranges[question_idx], 
                                                                                                original_token_lengths[question_idx], dense_layer)
    except:
        question_idx += 1
        break

    candidate_for_each_doc = []
    for i in range(character_positions.__len__()): # each candidate doc
        sentence_candidates = []
        for j in range(character_positions[i].__len__()): # each candidate sentence
            begin_position = character_positions[i][j][0]
            end_position = character_positions[i][j][-1]
            begin_index = sentence_indexes[i][j][0]
            end_index = sentence_indexes[i][j][1]
            score = min_distance_matrix[i][j]
            candidate = {
                "question_id": question_idx + 1,
                "sentence": candidate_documents[question_idx][i][begin_index:end_index], 
                "answer_begin_position ": begin_position,
                "answer_end_position": end_position,
                "article_id": candidate_document_ids[question_idx][i],
                "similarity_score": float(score)
            }
            sentence_candidates.append(candidate)
        candidate_for_each_doc.append(sentence_candidates)
    candidate_answers.append(candidate_for_each_doc)
    question_idx += 1

with open('./result/candidate_sentences_492.json', 'w', encoding='utf-8') as cand:
    try:
        json.dump(candidate_answers, cand, ensure_ascii=False)
    except:
        pass