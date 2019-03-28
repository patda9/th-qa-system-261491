import json
import numpy
import os
import re

from positive_generator import answers_detail, article_ids, MAX_SENTENCE_LENGTH, TKNED_DOCS_PATH
from positive_generator import preprocess_document, track_answer

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

    l_start_tk = answer_idx[0] - numpy.random.randint(10, 21)
    r_start_tk = answer_idx[-1] + numpy.random.randint(10, 21)
    l_step = 0
    r_step = 0
    
    temp_l_tk = []
    temp_l_char_range = []
    temp_l_index = []
    temp_r_tk = []
    temp_r_char_range = []
    temp_r_index = []

    try:
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
    samples_num = numpy.random.randint(2, 4)

    counter = 0
    negative1_samples = []
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

    return negative1_samples

if __name__ == "__main__":
    corpus_doc_ids = os.listdir(TKNED_DOCS_PATH)
    for i in range(len(corpus_doc_ids)):
        temp = ''
        for c in corpus_doc_ids[i]:
            if(c.isdigit()):
                temp += c
        corpus_doc_ids[i] = temp

    for i in range(len(article_ids)):
        current_doc = None
        with open('%s%s.json' % (TKNED_DOCS_PATH, article_ids[i]), encoding='utf-8-sig') as f:
            current_doc = json.load(f)
        negative0_samples, \
        negative0_samples_char_range, \
        negative0_samples_index = generate_from_answer_doc(answers_detail[i], current_doc)
        
        negative1_samples = generate_from_another_doc(corpus_doc_ids, article_ids[i])
        print(negative0_samples)
        print(negative1_samples)

        # print(negative_samples_char_range)
        # print(negative_samples_index)
        exit()