import json
import numpy
import os
import re

from pprint import pprint

answers_detail = None
with open('./data/new_sample_questions.json', encoding='utf-8-sig') as f:
    answers_detail = (json.load(f))['data']

article_ids = []
for ans in answers_detail:
    article_ids.append(ans['article_id'])

def fasttext_conversion(preprocessed_doc=None, vocab_vectors={}, enough_mem=False, limit=1000000):
    # fasttext_vec_file = open('C:/Users/Patdanai/Workspace/nlp-lab-27-2-2019/fastText/cc.th.300.vec', 'r', encoding='utf-8-sig')
    fasttext_vec_file = open('C:/Users/Patdanai/Downloads/wiki.th.vec', 'r', encoding='utf-8-sig')
    if(enough_mem):
        count = 0
        for line in fasttext_vec_file:
            if(count < 1):
                count = count + 1
                continue
            if(count < limit):
                line = line.split()
                vocab_vectors[line[0]] = line[1:]
            else:
                break
            count = count + 1
    else:
        temp = []
        i = 0
        while(i < len(preprocessed_doc)):
            try:
                if(preprocessed_doc[i] == ' ' or preprocessed_doc[i] == ';' or preprocessed_doc[i] == ''):
                    pass
                elif(preprocessed_doc[i].isdigit and preprocessed_doc[i+1] == '.' and preprocessed_doc[i+2].isdigit):
                    temp.append(preprocessed_doc[i] + preprocessed_doc[i+1] + preprocessed_doc[i+2])
                    i += 2
                else:
                    temp.append(preprocessed_doc[i])
            except:
                pass
            i += 1

        vocabs = set([w for w in temp])

        count = 0
        for line in fasttext_vec_file:
            if count > 0:
                line = line.split()
                if(line[0] in vocabs):
                    vocab_vectors[line[0]] = line[1:]
                elif(line[0] in vocab_vectors):
                    pass
            count = count + 1
        print('vocabs_num: %s' % len(vocab_vectors))

    return vocab_vectors

def preprocess_document(document):
    preprocess_doc = []
    for tk in document:
        pattern = re.compile(r"[<.*?>\"\'\n=!:“”&]|doc id=\"|url=|https://th|^wikipedia.org/(.*)|/doc")
        preprocessed_tk = re.sub(pattern, '', tk)
        preprocessed_tk = ''.join(c for c in preprocessed_tk if not(c in ['(', ')', '–', '_', ',', '-', ';', '{', '}']))
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

# tkned_th_wiki = os.listdir('../../tokenized-th-wiki')
TKNED_DOCS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'
MAX_SENTENCE_LENGTH = numpy.random.randint(60, 81)
OUTPUT_PATH = 'D:/Users/Patdanai/th-qasys-db/positive_sentences/'
OUTPUT_PATH_NPY = 'D:/Users/Patdanai/th-qasys-db/positive_embedded/'

from time import time

if __name__ == "__main__":
    start = time()
    # 485
    current_doc = None
    vocab_vectors = fasttext_conversion(enough_mem=True)
    end = time()
    et = end - start
    print(len(vocab_vectors))
    print('wvs loaded: %s seconds.' % et)

    for i in range(0, len(article_ids)):
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
        last_ans_tk = answer_idx[-1]
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
                    l_step -= 1
                elif(l_token is '' or l_token is ' '):
                    l_step -= 1
                else:
                    positive_sample.insert(0, l_token)
                    positive_sample_ans_masks.insert(0, l_token_mask)
                    positive_sample_char_range.insert(0, l_token_range)
                    positive_sample_index.insert(0, l_token_index)

                    l_count += 1
                    l_step -= 1
            except IndexError:
                positive_sample.insert(0, '<PAD>')
                l_count += 1
            
            try:
                r_token = preprocessed_doc[last_ans_tk + r_step]
                r_token_index = last_ans_tk + r_step
                r_token_mask = answer_masks[last_ans_tk + r_step]
                r_token_range = tokens_range[last_ans_tk + r_step]

                if(r_token is '' or r_token is ' '):
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
                r_step = len(preprocessed_doc) - last_ans_tk - r_step

        words_per_sample = 20
        fixed_words_num = words_per_sample
        embedded_sentences = []
        positive_samples = []
        start = positive_sample_index.index(first_ans_tk)
        for j in range(words_per_sample):
            try:
                if(start - j > -1):
                    sample = positive_sample[start - j:start + words_per_sample - j]
                    sample_index = positive_sample_index[start - j:start + words_per_sample - j]
                    sample_char_range = positive_sample_char_range[start - j:start + words_per_sample - j]
                    mask = [0] * fixed_words_num
                    mask[j] = 1
                else:
                    break
            except IndexError:
                print('here1')

            positive = {
                'article_id': article_ids[i], 
                'question_id': i + 1, 
                'sample_answer_maks': mask, 
                'sample_character_range': sample_char_range, 
                'sample_index': sample_index, 
                'sample_sentence': sample, 
            }
            positive_samples.append(positive)
        
            wvl = 300
            embedded_sentences
            word_vectors = numpy.zeros((len(sample), wvl))
            for k in range(len(sample)):
                try:
                    word_vectors[k, :] = vocab_vectors[sample[k]]
                except:
                    word_vectors[k, :] = word_vectors[k] 
                embedded_sentence = word_vectors
            embedded_sentences.append(embedded_sentence)

        out_file_name = '%spositive_question%s.json' % (OUTPUT_PATH, i)
        out_file_name_npy = '%spositive_question%s.npy' % (OUTPUT_PATH_NPY, i)

        with open(out_file_name, 'w', encoding='utf-8-sig', errors='ignore') as f:
            json.dump(positive_samples, f, ensure_ascii=False)

        numpy.save(out_file_name_npy, numpy.asarray(embedded_sentences))
        print(i, 'positive:', numpy.array(embedded_sentences).shape)
