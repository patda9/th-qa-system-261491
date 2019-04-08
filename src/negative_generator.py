import json
import numpy
import os
import re

from preprocessing import m_words_separate
from positive_generator import answers_detail, article_ids, MAX_SENTENCE_LENGTH, TKNED_DOCS_PATH
from positive_generator import fasttext_conversion, preprocess_document, track_answer

# negative 0
def generate_from_answer_doc(answer_detail, document):
    answer_masks, tokens_range = track_answer(answer_detail, document)
    preprocessed_doc = preprocess_document(document)

    negative0_samples = []
    negative0_samples_char_range = []
    negative0_samples_index = []
    
    answer_idx = []
    for j in range(len(answer_masks)):
        if(answer_masks[j]):
            answer_idx.append(j)

    l_start_tk = answer_idx[0] - numpy.random.randint(60, 81)
    r_start_tk = answer_idx[-1] + numpy.random.randint(60, 81)
    l_step = 0
    r_step = 0

    temp_l_tk = []
    temp_l_char_range = []
    temp_l_index = []
    temp_r_tk = []
    temp_r_char_range = []
    temp_r_index = []

    try:
        if(l_start_tk + l_step < 0):
            l_start_tk = answer_idx[0] // 2
        
        while(l_start_tk + l_step > -1):
            if(preprocessed_doc[l_start_tk + l_step] is ' ' or preprocessed_doc[l_start_tk + l_step] is ''):
                pass
            else:
                temp_l_tk.insert(0, preprocessed_doc[l_start_tk + l_step])
                temp_l_char_range.insert(0, tokens_range[l_start_tk + l_step])
                temp_l_index.insert(0, l_start_tk + l_step)
            l_step -= 1

        r_compensate = 0
        while(r_compensate > l_start_tk + l_step):
            r_compensate -= 1
        r_step += r_compensate
    except IndexError:
        print('l index error')

    try:
        if(r_start_tk + r_step > len(preprocessed_doc) - 1):
            r_start_tk = (len(preprocessed_doc) - 1 - answer_idx[-1]) // 2

        while(r_start_tk + r_step < len(preprocessed_doc)):
            if(preprocessed_doc[r_start_tk + r_step] is ' ' or preprocessed_doc[r_start_tk + r_step] is ''):
                pass
            else:
                temp_r_tk.append(preprocessed_doc[r_start_tk + r_step])
                temp_r_char_range.append(tokens_range[r_start_tk + r_step])
                temp_r_index.append(r_start_tk + r_step)
            r_step += 1

        l_compensate = len(preprocessed_doc)
        while(l_compensate < r_start_tk + r_step):
            l_compensate += 1
        l_step += l_compensate
    except IndexError:
        print('r index error')

    if(temp_l_tk):
        negative0_samples.append(temp_l_tk)
        negative0_samples_char_range.append(temp_l_char_range)
        negative0_samples_index.append(temp_l_index)
    if(temp_r_tk):
        negative0_samples.append(temp_r_tk)
        negative0_samples_char_range.append(temp_r_char_range)
        negative0_samples_index.append(temp_r_index)

    return negative0_samples, negative0_samples_char_range, negative0_samples_index

# negative 1
def generate_from_another_doc(corpus_doc_ids, article_id):
    answer_doc_id = article_id
    # samples_num = numpy.random.randint(2, 4)
    samples_num = 4

    counter = 0
    negative1_samples = []
    rand_doc_ids = []
    while(counter < samples_num):
        rand_index = numpy.random.randint(len(corpus_doc_ids))
        rand_doc_id = corpus_doc_ids[rand_index]
        while(answer_doc_id == rand_doc_id):
            rand_index = numpy.random.randint(len(corpus_doc_ids))
            rand_doc_id = corpus_doc_ids[rand_index]
        
        document = None
        with open('%s%s.json' % (TKNED_DOCS_PATH, rand_doc_id), encoding='utf-8-sig') as f:
            document = json.load(f)
        preprocessed_doc = preprocess_document(document)

        temp = []
        for tk in preprocessed_doc:
            if(tk is ' ' or tk is ''):
                pass
            else:
                temp.append(tk)
        counter += 1
        negative1_samples.append(temp)
        rand_doc_ids.append(rand_doc_id)

    return negative1_samples, rand_doc_ids

OUTPUT_PATH0 = 'D:/Users/Patdanai/th-qasys-db/negative_sentences/negative0/'
OUTPUT_PATH1 = 'D:/Users/Patdanai/th-qasys-db/negative_sentences/negative1/'
OUTPUT_PATH0_NPY = 'D:/Users/Patdanai/th-qasys-db/negative_embedded/negative0/'
OUTPUT_PATH1_NPY = 'D:/Users/Patdanai/th-qasys-db/negative_embedded/negative1/'

from time import time

words_per_sample = 20
if __name__ == "__main__":
    start = time()
    vocab_vectors = fasttext_conversion(enough_mem=True)
    end = time()
    et = end - start
    print(len(vocab_vectors))
    print('wvs loaded: %s seconds.' % et)

    corpus_doc_ids = os.listdir(TKNED_DOCS_PATH)
    for i in range(len(corpus_doc_ids)):
        temp = ''
        for c in corpus_doc_ids[i]:
            if(c.isdigit()):
                temp += c
        corpus_doc_ids[i] = temp

    negative0 = []
    for i in range(len(article_ids)):
        current_doc = None
        with open('%s%s.json' % (TKNED_DOCS_PATH, article_ids[i]), encoding='utf-8-sig') as f:
            current_doc = json.load(f)
        negative0_samples, \
        negative0_samples_char_range, \
        negative0_samples_index = generate_from_answer_doc(answers_detail[i], current_doc)
        negative0_samples = list(m_words_separate(words_per_sample, negative0_samples, overlapping_words=15)[0])
        negative0_samples = [list(s) for sentences in negative0_samples for s in sentences]
        negative0_samples_char_range = list(m_words_separate(words_per_sample, negative0_samples_char_range, overlapping_words=15)[0])
        negative0_samples_char_range = [list(s) for sentences in negative0_samples_char_range for s in sentences]
        negative0_samples_index = list(m_words_separate(words_per_sample, negative0_samples_index, overlapping_words=15)[0])
        negative0_samples_index = [list(s) for sentences in negative0_samples_index for s in sentences]

        negative = {
                'article_id': article_ids[i], 
                'question_id': i + 1, 
                'samples': list(negative0_samples), 
                'samples_char_range': list(negative0_samples_char_range), 
                'samples_index': list(negative0_samples_index)
        }

        wvl = 300
        embedded_sentences = []
        for j in range(len(negative0_samples)):
            word_vectors = numpy.zeros((len(negative0_samples[j]), wvl))
            for k in range(len(negative0_samples[j])):
                try:
                    word_vectors[k, :] = vocab_vectors[negative0_samples[j][k]]
                except:
                    word_vectors[k, :] = word_vectors[k] 
                embedded_sentence = word_vectors
            embedded_sentences.append(embedded_sentence)
        print(i, 'negative0:', numpy.array(embedded_sentences).shape)
    
        out_file_name0 = '%snegative0_question%s.json' % (OUTPUT_PATH0, i)
        out_file_name0_npy = '%snegative0_question%s.npy' % (OUTPUT_PATH0_NPY, i)
        with open(out_file_name0, 'w', encoding='utf-8-sig', errors='ignore') as f:
            json.dump(negative, f, ensure_ascii=False)
        numpy.save(out_file_name0_npy, numpy.asarray(embedded_sentences))

        negative1_samples, rand_doc_ids = generate_from_another_doc(corpus_doc_ids, article_ids[i])
        negative1_samples = list(m_words_separate(words_per_sample, negative1_samples, overlapping_words=15)[0])
        negative1_samples = [list(s) for sentences in negative1_samples for s in sentences]
        negative = {
                'article_ids': rand_doc_ids, 
                'question_id': i + 1, 
                'samples': negative1_samples, 
        }

        embedded_sentences = []
        for j in range(len(negative1_samples)):
            word_vectors = numpy.zeros((len(negative1_samples[j]), wvl))
            for k in range(len(negative1_samples[j])):
                try:
                    word_vectors[k, :] = vocab_vectors[negative1_samples[j][k]]
                except:
                    word_vectors[k, :] = word_vectors[k] 
                embedded_sentence = word_vectors
            embedded_sentences.append(embedded_sentence)
        print(i, 'negative1:', numpy.array(embedded_sentences).shape)

        out_file_name1 = '%snegative1_question%s.json' % (OUTPUT_PATH1, i)
        out_file_name1_npy = '%snegative1_question%s.npy' % (OUTPUT_PATH1_NPY, i)
        with open(out_file_name1, 'w', encoding='utf-8-sig', errors='ignore') as f:
            json.dump(negative, f, ensure_ascii=False)
        numpy.save(out_file_name1_npy, numpy.asarray(embedded_sentences))
