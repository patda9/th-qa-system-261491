import json
import os
import re

def preprocess_document(document):
    preprocess_doc = []
    for tk in document:
        pattern = re.compile(r"[<.*?>\"\'\n=!:]|doc id=\"|url=|https://th|^wikipedia.org/(.*)|/doc")
        preprocessed_tk = re.sub(pattern, '', tk)
        preprocessed_tk = ''.join(c for c in preprocessed_tk if not(c in ['(', ')', '–', '_', ',', '-']))
        preprocess_doc.append(preprocessed_tk)
    return preprocess_doc

def track_answer(answers_detail, document):
    answer_masks = []
    tokens_range = []

    counter = 0
    end = 0
    start = 0
    for tk in document:
        end += len(tk)
        characters_index = range(start, end)

        ans_begin = answers_detail[i]['answer_begin_position ']
        ans_end = answers_detail[i]['answer_end_position']
        if(ans_begin - 1 in characters_index or ans_end - 1 in characters_index):
            answer_masks.append(1)
        else:
            answer_masks.append(0)

        counter += 1
        start = end

        tokens_range.append(characters_index)
    
    return answer_masks, tokens_range

answers_detail = None
with open('../new_sample_questions.json', encoding='utf-8-sig') as f:
    answers_detail = (json.load(f))['data']

# data = None
# with open('../665.json', encoding='utf-8-sig') as f:
#     data = json.load(f)

article_ids = []
for ans in answers_detail:
    article_ids.append(ans['article_id'])

# tkned_th_wiki = os.listdir('../../tokenized-th-wiki')

MAX_SENTENCE_LENGTH = 60

if __name__ == "__main__":
    current_doc = None
    for i in range(len(article_ids)):
        with open('../../tokenized-th-wiki/%s.json' % (article_ids[i]), encoding='utf-8-sig') as f:
            current_doc = json.load(f)

        answer_masks, tokens_range = track_answer(answers_detail, current_doc)
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
                pass
            
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
                pass

        # for output testing
        print(positive_sample)
        print(positive_sample_char_range)
        print(positive_sample_index)
        print(positive_sample_ans_masks)
        # print(answer_masks)
        # print(tokens_range)
        # print(current_doc[151])
        # print(tokens_range.index([528, 529, 530, 531, 532, 533, 534, 535, 536]))
        exit()
