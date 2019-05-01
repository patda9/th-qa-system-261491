import json
import numpy as np
import os
import re

from preprocessing import m_words_separate
from positive_generator import answers_detail, article_ids, MAX_SENTENCE_LENGTH
from positive_generator import preprocess_document, track_answer

np.random.seed(0)

def get_vocab_wvs(wv_path, preprocessed_doc=None, vocabs=None):
    fasttext_fp = open(wv_path, encoding='utf-8-sig')
    white_spaces = ['', ' ']
    
    if(not(vocabs) and preprocessed_doc):
        vocabs = set([tk for tk in preprocessed_doc if tk not in white_spaces])

    vocab_wvs = {}

    line_count = 0
    vocab_count = 0
    for line in fasttext_fp:
        if(line_count > 0):
            line = line.split()
            if(vocab_count < len(vocabs)):
                if(line[0] in vocabs):
                    vocab_wvs[line[0]] = line[1:]
                    print('found %s %s total_len: %s' % (line_count, line[0], len(vocabs)))
                    vocab_count += 1
                    print(vocab_count)
            else:
                break
        line_count += 1
    
    return vocab_wvs

def vectorize_tokens(sentence, vocab_wvs=None, wvl=300):
    word_vectors = np.zeros((len(sentence), wvl))
    for i in range(len(sentence)):
        try:
            if(sentence[i] != '<PAD>'):
                word_vectors[i, :] = vocab_wvs[sentence[i]]
        except:
            pass

    return word_vectors

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

        l_start_tk = 0
        r_start_tk = len(preprocessed_doc)
    try:
        l_start_tk = answer_idx[0] - np.random.randint(60, 81)
        r_start_tk = answer_idx[-1] + np.random.randint(60, 81)
    except:
        pass

    l_step = 0
    r_step = 0

    temp_l_tk = []
    temp_l_char_range = []
    temp_l_index = []
    temp_r_tk = []
    temp_r_char_range = []
    temp_r_index = []

    try:
        if(l_start_tk + l_step < 1):
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
        l_step = len(preprocessed_doc) // 2
        while(l_start_tk > -1):
            if(preprocessed_doc[l_step] is ' ' or preprocessed_doc[l_step] is ''):
                pass
            else:
                temp_l_tk.insert(0, preprocessed_doc[l_step])
                temp_l_char_range.insert(0, tokens_range[l_step])
                temp_l_index.insert(0, l_step)
            if(l_step > 0):
                l_step -= 1
            else:
                break

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
        r_start_tk = len(preprocessed_doc) // 2 + 1

        while(r_start_tk + r_step < len(preprocessed_doc)):
            if(preprocessed_doc[r_start_tk + r_step] is ' ' or preprocessed_doc[r_start_tk + r_step] is ''):
                pass
            else:

                temp_r_tk.append(preprocessed_doc[r_start_tk + r_step])
                temp_r_char_range.append(tokens_range[r_start_tk + r_step])
                temp_r_index.append(r_start_tk + r_step)
            r_step += 1

    if(l_step + r_step > len(preprocessed_doc) - 1):
        while(len(temp_l_tk) < words_per_sample):
            temp_l_tk.insert(0, '<PAD>')
            temp_l_char_range.insert(0, (-1, -1))
            temp_l_index.insert(0, -1)
        while(len(temp_r_tk) < words_per_sample):
            temp_r_tk.insert(0, '<PAD>')
            temp_r_char_range.insert(0, (-1, -1))
            temp_r_index.insert(0, -1)

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
    samples_num = 32

    counter = 0
    negative1_samples = []
    rand_doc_ids = []
    while(counter < samples_num):
        rand_index = np.random.randint(len(corpus_doc_ids))
        rand_doc_id = corpus_doc_ids[rand_index]
        while(answer_doc_id == rand_doc_id):
            rand_index = np.random.randint(len(corpus_doc_ids))
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

# OUTPUT_PATH0 = 'D:/Users/Patdanai/th-qasys-db/negative_sentences/negative0/'
OUTPUT_PATH0 = 'C:/Users/Patdanai/Desktop/492/negative0/tokenized/'
# OUTPUT_PATH1 = 'D:/Users/Patdanai/th-qasys-db/negative_sentences/negative1/'
OUTPUT_PATH1 = 'C:/Users/Patdanai/Desktop/492/negative1/tokenized/'
# OUTPUT_PATH0_NPY = 'D:/Users/Patdanai/th-qasys-db/negative_embedded/negative0/'
OUTPUT_PATH0_NPY = 'C:/Users/Patdanai/Desktop/492/negative0/embedded/'
# OUTPUT_PATH1_NPY = 'D:/Users/Patdanai/th-qasys-db/negative_embedded/negative1/'
OUTPUT_PATH1_NPY = 'C:/Users/Patdanai/Desktop/492/negative1/embedded/'

# TKNED_DOCS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'
TKNED_DOCS_PATH = 'C:/Users/Patdanai/Desktop/tokenized-th-wiki/'
words_per_sample = 40
wv_path = 'C:/Users/Patdanai/Desktop/261499-nlp/lab/cc.th.300.vec'
if __name__ == "__main__":
    corpus_doc_ids = os.listdir(TKNED_DOCS_PATH)
    for i in range(len(corpus_doc_ids)):
        temp = ''
        for c in corpus_doc_ids[i]:
            if(c.isdigit()):
                temp += c
        corpus_doc_ids[i] = temp

    batch_size = 4000 # default: len(article_ids)
    start = 0 # index

    batch_vocabs = []
    for i in range(start, batch_size + start):
        with open('%s%s.json' % (TKNED_DOCS_PATH, article_ids[i]), encoding='utf-8-sig') as f:
            current_doc = json.load(f)

        preprocessed_doc = preprocess_document(current_doc)
        batch_vocabs += preprocessed_doc

    batch_vocabs = set(batch_vocabs)
    batch_vocabs.remove('')

    vocab_wvs = get_vocab_wvs(wv_path, vocabs=batch_vocabs)

    for i in range(start, batch_size + start):
        current_doc = None
        with open('%s%s.json' % (TKNED_DOCS_PATH, article_ids[i]), encoding='utf-8-sig') as f:
            current_doc = json.load(f)
        preprocessed_doc = preprocess_document(current_doc)
        current_doc = preprocessed_doc

        negative0_samples, \
        negative0_samples_char_range, \
        negative0_samples_index = generate_from_answer_doc(answers_detail[i], current_doc)
        negative0_samples = list(m_words_separate(words_per_sample, negative0_samples, overlapping_words=15)[0])[0]
        negative0_samples_char_range = list(m_words_separate(words_per_sample, negative0_samples_char_range, overlapping_words=15)[0])
        negative0_samples_char_range = [list(s) for sentences in negative0_samples_char_range for s in sentences]
        negative0_samples_index = list(m_words_separate(words_per_sample, negative0_samples_index, overlapping_words=15)[0])
        negative0_samples_index = [list(s) for sentences in negative0_samples_index for s in sentences]

        n0_sampling_num = 10
        if(n0_sampling_num < 1):
            n0_sampling_num += 1
        negative0_samples = np.random.permutation(negative0_samples)[:n0_sampling_num]
        negative0_samples = [list(s) for s in negative0_samples]

        negative = {
                'article_id': article_ids[i], 
                'question_id': i + 1, 
                'samples': list(negative0_samples), 
                'samples_char_range': list(negative0_samples_char_range), 
                'samples_index': list(negative0_samples_index)
        }
        
        embedded_sentences = []
        for j in range(len(negative0_samples)):
            es = vectorize_tokens(negative0_samples[j], vocab_wvs=vocab_wvs)
            embedded_sentences.append(es)
        
        out_file_name0 = '%snegative0_question%s.json' % (OUTPUT_PATH0, i)
        out_file_name0_npy = '%snegative0_question%s.npy' % (OUTPUT_PATH0_NPY, i)
        with open(out_file_name0, 'w', encoding='utf-8-sig', errors='ignore') as f:
            json.dump(negative, f, ensure_ascii=False)
        np.save(out_file_name0_npy, np.array(embedded_sentences))
        
        print(i, 'negative0:', np.array(embedded_sentences).shape)

    batch_size = 10 # default: len(article_ids)
    start = 0 # index

    negative1_batch, doc_id_batch = [], []
    for i in range(start, batch_size + start):
        negative1_samples, rand_doc_ids = generate_from_another_doc(corpus_doc_ids, article_ids[i])
        negative1_samples = list(m_words_separate(words_per_sample, negative1_samples, overlapping_words=15)[0])

        n1s = []
        for j in range(len(negative1_samples)):
            try:
                r = np.random.randint(len(negative1_samples[j]))
                if(negative1_samples[j][r]):
                    n1s.append(negative1_samples[j][r])
            except:
                pass

        negative1_samples = n1s
        
        negative1_batch.append(negative1_samples)
        doc_id_batch += rand_doc_ids

        negative = {
                'article_ids': rand_doc_ids, 
                'question_id': i + 1, 
                'samples': negative1_samples, 
        }

        out_file_name1 = '%snegative1_question%s.json' % (OUTPUT_PATH1, i)
        with open(out_file_name1, 'w', encoding='utf-8-sig', errors='ignore') as f:
            json.dump(negative, f, ensure_ascii=False)

    doc_id_batch = list(set(doc_id_batch))

    batch_vocabs = []
    for i in range(start, batch_size + start):
        with open('%s%s.json' % (TKNED_DOCS_PATH, doc_id_batch[i]), encoding='utf-8-sig') as f:
            current_doc = json.load(f)

        preprocessed_doc = preprocess_document(current_doc)
        batch_vocabs += preprocessed_doc

    batch_vocabs = set(batch_vocabs)
    batch_vocabs.remove('')
    
    vocab_wvs = get_vocab_wvs(wv_path, vocabs=batch_vocabs)
    
    es_batch = []
    for i in range(len(negative1_batch)):
        embedded_sentences = []
        for j in range(len(negative1_batch[i])):
            es = vectorize_tokens(negative1_samples[i], vocab_wvs)
            embedded_sentences.append(es)
        es_batch.append(es)
        
        out_file_name1_npy = '%snegative1_question%s.npy' % (OUTPUT_PATH1_NPY, i)
        np.save(out_file_name1_npy, np.asarray(embedded_sentences))
        print(i, 'negative1:', np.array(embedded_sentences).shape)
