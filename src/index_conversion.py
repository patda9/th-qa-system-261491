import json
import numpy as np
import os

DOCUMENTS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'

def write_correct_sentences(DOCUMENTS_PATH):
    correct_document_ids = []
    correct_question_ids = []
    correct_sentences_index = []

    with open('./results/correct_sentences_index/correct_sentences_index.json', 'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            correct_document_ids.append(data[i]['article_id'])
            correct_question_ids.append(data[i]['question_id'])
            correct_sentences_index.append(data[i]['sentence_index'])
        
    print(correct_document_ids[0])
    print(correct_question_ids[0] + 1)
    print(correct_sentences_index[0])

    cs_tokens = []
    question_no = 1
    for doc_f in correct_document_ids:
        with open(DOCUMENTS_PATH + doc_f + '.json', 'r', encoding='utf-8', errors='ignore') as f:
            doc = json.load(f)
            begin_index, end_index = correct_sentences_index[question_no-1][0], correct_sentences_index[question_no-1][-1] + 1
            cs_tokens.append({
                "article_id": doc_f, 
                "question_id": correct_question_ids[question_no-1], 
                "sentence_tokens":doc[begin_index:end_index]
            })
        question_no += 1

    with open('./results/correct_sentences_tokens/correct_sentences_tokens.json', 'w', encoding='utf-8', errors='ignore') as f:
        data = json.dump(cs_tokens, f, indent=4)

    with open('./results/correct_sentences_tokens/correct_sentences_tokens_readable.json', 'w', encoding='utf-8', errors='ignore') as f:
        data = json.dump(cs_tokens, f, ensure_ascii=False, indent=4)

def write_incorrect_sentences(DOCUMENTS_PATH):
    incorrect_document_ids = []
    incorrect_question_ids = []
    incorrect_sentences_index = []
    incorrect_sentences_ranks = []

    with open('./results/incorrect_sentences_index/incorrect_sentences_index.json', 'r') as f:
        data = json.load(f)

        for i in range(len(data)):
            try:
                incorrect_document_ids.append(data[i][0]['article_id'])
                incorrect_question_ids.append(data[i][0]['question_id'])
                incorrect_sentences_index.append(data[i][0]['sentence_index'])
            except:
                pass
    print(incorrect_document_ids.__len__())
    print(incorrect_question_ids.__len__())
    print(np.array(incorrect_sentences_index))

    incs_tokens = []
    question_no = 1
    for doc_f in incorrect_document_ids:
        with open(DOCUMENTS_PATH + doc_f + '.json', 'r', encoding='utf-8', errors='ignore') as f:
            doc = json.load(f)
            begin_index, end_index = incorrect_sentences_index[question_no-1][0], incorrect_sentences_index[question_no-1][-1] + 1
            incs_tokens.append({
                "article_id": doc_f, 
                "question_id": incorrect_question_ids[question_no-1], 
                "sentence_tokens":doc[begin_index:end_index]
            })
        question_no += 1

    with open('./results/incorrect_sentences_tokens/incorrect_sentences_tokens.json', 'w', encoding='utf-8', errors='ignore') as f:
        data = json.dump(incs_tokens, f, indent=4)

    with open('./results/incorrect_sentences_tokens/incorrect_sentences_tokens_readable.json', 'w', encoding='utf-8', errors='ignore') as f:
        data = json.dump(incs_tokens, f, ensure_ascii=False, indent=4)

write_incorrect_sentences(DOCUMENTS_PATH)