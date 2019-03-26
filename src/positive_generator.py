import json
import os

answer_details = None
with open('../new_sample_questions.json', encoding='utf-8-sig') as f:
    answer_details = (json.load(f))['data']

data = None
# with open('../665.json', encoding='utf-8-sig') as f:
#     data = json.load(f)

article_ids = []
for ans in answer_details:
    article_ids.append(ans['article_id'])

# print(article_ids)

# tkned_th_wiki = os.listdir('../../tokenized-th-wiki')
for i in range(len(article_ids)):
    start = 0
    end = 0

    with open('../../tokenized-th-wiki/%s.json' % (article_ids[i]), encoding='utf-8-sig') as f:
        current_doc = json.load(f)
    
    answer_masks = []
    counter = 0
    tokens_range = []
    for tk in current_doc:
        end += len(tk)
        characters_index = list(range(start, end))

        ans_begin = answer_details[i]['answer_begin_position ']
        ans_end = answer_details[i]['answer_end_position']
        if(ans_begin - 1 in characters_index or ans_end - 1 in characters_index):
            answer_masks.append(1)
            print(current_doc[counter])
        else:
            answer_masks.append(0)

        counter += 1
        
        start = end
        tokens_range.append(characters_index)

    # print(answer_masks)
    # print(current_doc[151])
    # print(tokens_range.index([528, 529, 530, 531, 532, 533, 534, 535, 536]))
    exit()
