import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import preprocessing as prep
import re
import sentence_vectorization as sv
import os

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from pprint import pprint

np.random.seed(0)

if(__name__ == '__main__'):
    tokens_path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/' # get tokenized articles content
    plain_text_path = 'C:/Users/Patdanai/Desktop/documents-nsc/' # get plain text article content
    tokens_dataset = os.listdir(tokens_path)
    n_samples = 5 # number of samples from nsc questions

    models_ws_archs_path = 'C:/Users/Patdanai/Desktop/492/th-qa-system-261491/models'
    model_files = os.listdir(models_ws_archs_path)

    nsc_sample_questions = []
    with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8', errors='ignore') as f:
        nsc_sample_questions = json.load(f)
    
    nsc_answer_doc_id = []
    with open('./../new_sample_questions_answer.json', 'r', encoding='utf-8', errors='ignore') as f:
        nsc_answer_doc_id = json.load(f)

    nsc_answer_details = {}
    with open('./../new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        nsc_answer_details = json.load(f)
    
    # get first n samples from questions
    count = 0
    last_doc_id = 282972
    selected_article_ids = []
    selected_questions_numbers = []
    selected_plain_text_questions = []
    for q in nsc_answer_details['data']:
        if(q['article_id'] < last_doc_id and count < n_samples): # limitted preprocessed docs id: 282972
            selected_article_ids.append(q['article_id'])
            selected_questions_numbers.append(q['question_id'])
            selected_plain_text_questions.append(q['question'])
            count += 1

    # print(np.asarray(list(zip(selected_questions_numbers, selected_article_ids)))) # TESTING FUNCTION: map question numbers to article ids

    """
    output: array of tokens <arrays of tokenized article content: array like>
    input: path of tokens <file path: string>
    """
    # load tokens from each article
    selected_plain_text_article = []
    selected_tokenized_articles = [] # array for arrays of tokens
    for i in selected_article_ids:
        article_path = os.path.join(tokens_path, str(i) + '.json')
        plain_text_article_path = os.path.join(plain_text_path, str(i) + '.txt')
        with open(plain_text_article_path, 'r', encoding='utf-8', errors='ignore') as f:
            plain_text_article = f.read()
        selected_plain_text_article.append(plain_text_article)
        with open(article_path, 'r', encoding='utf-8', errors='ignore') as f:
            article_tokens = json.load(f)
        selected_tokenized_articles.append(article_tokens)

    # print(np.asarray(selected_tokenized_articles)) # TESTING FUNCTION: arrays of article tokens

    """
    output: preprocessed questions tokens <arrays of tokenized questions: array like>
    input: questions tokens <arrays of tokenized questions: array like>
    """
    # remove noise from tokenized questions
    selected_tokenized_questions = []
    for i in range(nsc_sample_questions.__len__()):
        if((i+1) in selected_questions_numbers): # use (i+1) as index: first question starts at 1 
            nsc_sample_questions[i] = [w for w in nsc_sample_questions[i] if not(w is ' ')]
            selected_tokenized_questions.append(nsc_sample_questions[i])

    # print(np.asarray(selected_tokenized_questions)) # TESTING FUNCTION: arrays of questions tokens
    
    remaining_tokens = []
    original_tokens_indexes = []
    original_tokens_ranges = []
    j = 1
    for ids in selected_article_ids:
        file_path = os.path.join(tokens_path, str(ids) + '.json')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            preprocessed_article = prep.remove_noise(data)
        remaining_tokens.append(preprocessed_article[0])
        original_tokens_indexes.append(preprocessed_article[1])
        original_tokens_ranges.append(preprocessed_article[2])
        j += 1 

    print(np.asarray(remaining_tokens[0])) # TESTING FUNCTION: arrays of preprocessed tokens,
    # and also => preprocessed tokens indexes
    # and also => and preprocessed tokens ranges
    print(np.asarray(original_tokens_indexes[0]))
    print(np.asarray(list(zip(original_tokens_indexes[0], remaining_tokens[0])))) # TESTING FUNCTION: arrays of preprocessed tokens
    # and preprocessed tokens indexes
    print(np.asarray(original_tokens_ranges[0])) # each before-preprocessing token's ending position

    # # create vocabularies from input articles
    # vocabularies = [article[i] for article in remaining_tokens for i in range(article.__len__())]
    # # create word to id dictionary
    # word2id = {}
    # for (i, w) in enumerate(set(vocabularies)):
    #     try:
    #         word2id[w] = i
    #     except ValueError:
    #         pass
    # # create word_id to word dictionary
    # id2word = {idx: w for w, idx in word2id.items()}
    
    # # pprint(word2id) # TESTING FUNCTION: dict of words: ids
    # # pprint(id2word) # TESTING FUNCTION: dict of ids: words

    """
    output: 
    input: 
    """
    # preprocess questions
    preprocessed_questions = []
    for question in selected_tokenized_questions:
        temp = prep.remove_noise(question)[0]
        preprocessed_questions.append(temp)
    
    print(np.asarray(preprocessed_questions))
    
    # selected_question_id_representation = []
    # for q in selected_questions:
    #     temp = []
    #     for w in q:
    #         try:
    #             temp.append(word2id[w])
    #         except KeyError:
    #             pass
    #     selected_question_id_representation.append(temp)

    # # print(selected_question_id_representation)
    # # print(selected_articles[-1])

    # content_id_representation = []
    # for q in remaining_words:
    #     temp = []
    #     for w in q:
    #         try:
    #             temp.append(word2id[w[1]])
    #         except KeyError:
    #             pass
    #     content_id_representation.append(temp)

    words_per_sentence = 20
    overlapping_words = words_per_sentence // 2

    m_words_preprocessed_article = sv.m_words_separate(words_per_sentence, remaining_tokens, overlapping_words=overlapping_words)

    print(np.asarray(m_words_preprocessed_article[0]))
    print(np.asarray(m_words_preprocessed_article[1]))