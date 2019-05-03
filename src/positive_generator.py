import json
import numpy as np
import os
import re

from pprint import pprint

answers_detail = None
with open('./data/nsc_questions_answers.json', encoding='utf-8-sig') as f:
    answers_detail = (json.load(f))['data']

article_ids = []
for ans in answers_detail:
    article_ids.append(ans['article_id'])

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

def preprocess_document(document):
    preprocess_doc = []
    for tk in document:
        sp_url_pattern = re.compile(r"[\"#$%&\'()*+,-/:;<=>?@[\\\]^_`{\|}~“”!]|doc id=\"|url=|^https:(.*)|^wikipedia.org(.*)|\\u(.*)")
        doc_pattern = re.compile(r"doc(.|[\n]*)")
        acronym_pattern = re.compile(r"(([a-zA-Z\u0e00-\u0ef70-9]+[.])*[a-zA-Z\u0e00-\u0ef70-9]*)") # พ.ศ. ดร.

        if(re.fullmatch(acronym_pattern, tk)):
            preprocessed_tk = tk
        else:
            preprocessed_tk = re.sub(sp_url_pattern, '', tk)
            preprocessed_tk = re.sub(doc_pattern, '', preprocessed_tk)

        preprocessed_tk = ''.join(c for c in preprocessed_tk if not(c in ['(', ')', '–', '_', ',', '-', ';', '{', '}', ' ']))
        preprocess_doc.append(preprocessed_tk)

    return preprocess_doc

def track_answer(answer_detail, document):
    answer_masks = []
    tokens_range = []

    counter = 0
    end = 0
    start = 0
    for tk in document:
        end += len(tk)
        characters_index = (start, end)
        
        ans_begin = answer_detail['answer_begin_position ']
        ans_end = answer_detail['answer_end_position']
        if(ans_begin - 1 in range(start, end) or ans_end - 1 in range(start, end)):
            answer_masks.append(1)
        else:
            answer_masks.append(0)

        counter += 1
        start = end

        tokens_range.append(characters_index)
    
    return answer_masks, tokens_range

def vectorize_tokens(sentence, vocab_wvs=None, wvl=300):
    word_vectors = np.zeros((len(sentence), wvl))
    for i in range(len(sentence)):
        try:
            if(sentence[i] != '<PAD>'):
                word_vectors[i, :] = vocab_wvs[sentence[i]]
        except:
            pass

    return word_vectors

TKNED_DOCS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'
TKNED_DOCS_PATH = 'C:/Users/Patdanai/Desktop/tokenized-th-wiki/'
MAX_SENTENCE_LENGTH = 80
OUTPUT_PATH = 'D:/Users/Patdanai/th-qasys-db/positive_sentences/'
OUTPUT_PATH = 'C:/Users/Patdanai/Desktop/492/positive/positive_tokenized/'
OUTPUT_PATH_NPY = 'D:/Users/Patdanai/th-qasys-db/positive_embedded/'
OUTPUT_PATH_NPY = 'C:/Users/Patdanai/Desktop/492/positive/positive_embedded/'

wv_path = 'C:/Users/Patdanai/Desktop/261499-nlp/lab/cc.th.300.vec'
if __name__ == "__main__":
    # put wvs to memory
    batch_size = 4000
    start = 0

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
        with open('%s%s.json' % (TKNED_DOCS_PATH, article_ids[i]), encoding='utf-8-sig') as f:
            current_doc = json.load(f)

        answer_masks, tokens_range = track_answer(answers_detail[i], current_doc)
        preprocessed_doc = preprocess_document(current_doc)

        answer_idx = []
        for j in range(len(answer_masks)):
            if(answer_masks[j]):
                answer_idx.append(j)

        positive_sample = []
        positive_sample_ans_masks = []
        positive_sample_char_range = []
        positive_sample_index = []        

        first_ans_tk = answer_idx[0]
        if(preprocessed_doc[first_ans_tk] in ['', ' ']):
            first_ans_tk = last_ans_tk
        
        last_ans_tk = answer_idx[-1]
        if(preprocessed_doc[last_ans_tk] in ['', ' ']):
            last_ans_tk = first_ans_tk

        l_count = 0
        r_count = 0
        l_step = 0
        r_step = 0
        while(l_count + r_count < MAX_SENTENCE_LENGTH and l_count + r_count < len(preprocessed_doc)):
            try:
                l_token = preprocessed_doc[first_ans_tk + l_step]
                l_token_index = first_ans_tk + l_step
                l_token_mask = answer_masks[first_ans_tk + l_step]
                l_token_range = tokens_range[first_ans_tk + l_step]

                if(first_ans_tk + l_step < 0):
                    l_step = 0
                elif(l_token in ['', ' ']):
                    l_step -= 1
                else:
                    positive_sample.insert(0, l_token)
                    positive_sample_ans_masks.insert(0, l_token_mask)
                    positive_sample_char_range.insert(0, l_token_range)
                    positive_sample_index.insert(0, l_token_index)
                    l_count += 1
                    l_step -= 1

            except IndexError:
                l_count += 1

            try:
                r_token = preprocessed_doc[last_ans_tk + r_step]
                r_token_index = last_ans_tk + r_step
                r_token_mask = answer_masks[last_ans_tk + r_step]
                r_token_range = tokens_range[last_ans_tk + r_step]
                if(last_ans_tk + r_step > len(preprocessed_doc) - 1):
                    pass
                elif(r_token in ['', ' ']):
                    r_step += 1
                else:
                    positive_sample.append(r_token)
                    positive_sample_ans_masks.append(r_token_mask)
                    positive_sample_char_range.append(r_token_range)
                    positive_sample_index.append(r_token_index)

                    r_count += 1
                    r_step += 1
            except IndexError:
                l_step += len(preprocessed_doc) - last_ans_tk - r_step
                r_step += 1
        
        words_per_sample = 40
        sample_num = 10
        embedded_sentences = []
        positive_samples = []
        start_idx = positive_sample_index.index(first_ans_tk) - words_per_sample // 2 - sample_num // 2
        for j in range(0, sample_num, 2):
            try:
                if(start_idx - j > -1 and start_idx + words_per_sample - j < len(positive_sample) - 1):
                    sample = positive_sample[start_idx - j:start_idx + words_per_sample - j]
                    sample_index = positive_sample_index[start_idx - j:start_idx + words_per_sample - j]
                    sample_char_range = positive_sample_char_range[start_idx - j:start_idx + words_per_sample - j]
                    mask = [0] * words_per_sample

                    if(last_ans_tk - first_ans_tk > 0):
                        for k in range(positive_sample_index.index(first_ans_tk) - start_idx + j, positive_sample_index.index(last_ans_tk) - start_idx + j):
                            mask[k] = 1
                    else:
                        mask[positive_sample_index.index(first_ans_tk) - start_idx + j] = 1
                else:
                    sample = positive_sample[:]
                    sample_index = positive_sample_index[:]
                    sample_char_range = positive_sample_char_range[:]
                    mask = [0] * len(positive_sample_index)

                    if(len(sample) > words_per_sample):
                        if(last_ans_tk - first_ans_tk > 0):
                            for k in range(positive_sample_index.index(first_ans_tk), positive_sample_index.index(last_ans_tk)):
                                mask[k] = 1
                        else:
                            mask[positive_sample_index.index(first_ans_tk)] = 1

                        l_removal, r_removal = 0, len(sample)
                        while(len(sample) > words_per_sample):
                            try:
                                temp_first = positive_sample_index.index(first_ans_tk)
                                temp_last = positive_sample_index.index(last_ans_tk)
                            except:
                                temp_last = temp_first
                            
                            if(l_removal - 1 < r_removal - temp_last):
                                sample.pop()
                                sample_index.pop()
                                sample_char_range.pop()
                                mask.pop()
                                r_removal -= 1
                            elif(r_removal < l_removal - 1):
                                sample.pop(0)
                                sample_index.pop(0)
                                sample_char_range.pop(0)
                                mask.pop(0)
                                l_removal += 1
                            elif(r_removal - temp_last > 0):
                                sample.pop()
                                sample_index.pop()
                                sample_char_range.pop()
                                mask.pop()
                                r_removal -= 1
                            else:
                                sample.pop(0)
                                sample_index.pop(0)
                                sample_char_range.pop(0)
                                mask.pop(0)
                                l_removal += 1
                    else:
                        sample_ans_mask = positive_sample_ans_masks[:]
                        while(len(sample) < words_per_sample):
                            sample.insert(0, '<PAD>')
                            sample_ans_mask.insert(0, 0)
                            sample_char_range.insert(0, (-1, -1))
                            sample_index.insert(0, -1)
            except IndexError:
                exit('Index Error: from line 276')

            positive = {
                'article_id': article_ids[i], 
                'question_id': i + 1, 
                'sample_answer_maks': mask, 
                'sample_character_range': sample_char_range, 
                'sample_index': sample_index, 
                'sample_sentence': sample, 
            }
            positive_samples.append(positive)

            es = vectorize_tokens(sample, vocab_wvs=vocab_wvs)
            embedded_sentences.append(es)

        out_file_name = '%spositive_question%s.json' % (OUTPUT_PATH, i)
        out_file_name_npy = '%spositive_question%s.npy' % (OUTPUT_PATH_NPY, i)

        with open(out_file_name, 'w', encoding='utf-8-sig', errors='ignore') as f:
            json.dump(positive_samples, f, ensure_ascii=False)

        np.save(out_file_name_npy, np.asarray(embedded_sentences))
        print(i, 'positive:', np.array(embedded_sentences).shape)
        print()
