"""
plot histogram of true answer from json output
    foreach nsc_answer_detail
        get answer token ranges
        locate answer index(token range)
        check if candidate answer sentence range is overlap with nsc answer token range
    plot histogram 
"""

import candidates_sentences_selection as cs
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sentence_vectorization as sv

from matplotlib import colors
from matplotlib.patches import Rectangle 
from matplotlib.ticker import PercentFormatter
from pprint import pprint

def plot_histogram(path, ranks, bins_num=1, legends=[False, [], 'best'], x_label='', y_label=''): # bins = c_sentence length
    fig1, axs1 = plt.subplots(1, tight_layout=True)
    n, bins, patches = axs1.hist(ranks, bins=bins_num)
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    # title = str(words_per_sentence) + '-Words Sentence (with ' + str(overlapping_words//2) + ' words overlapped) Model ' + str(n_samples) + ' Samples'
    # fig1.suptitle('True answer similarity ranks', fontsize=12, fontweight='bold')
    # axs1.set_xlabel('True answer sentence Rank <N-TH>')
    if(legends[0]):
        cmap = plt.get_cmap('jet')
        low = cmap(0.5)
        medium =cmap(0.25)
        high = cmap(0.8)
        handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low, medium, high]]
        labels = legends[1]
        plt.legend(handles, labels)
    axs1.set_xlabel(x_label)
    axs1.set_ylabel(y_label)
    plt.savefig(path)
    plt.show()

# check if candidate sentences are answer or not and what is their rank (less euclidean more rank)
def check_origin(answer_details, candidate_answers):
    doc_not_found = 0
    max_c_num = 8
    ans_not_found_question = []
    doc_not_found_question = []
    has_ans_question = []
    true_ans_ranks = []
    
    for i in range(len(candidate_answers)):
        ans_doc_id = answer_details[i]['article_id']
        ans_pos_range = (answer_details[i]['answer_begin_position '], answer_details[i]['answer_end_position'])
        k = 1 # c_doc number
        for c_doc in candidate_answers[i]:
            # print(c_rank)
            # exit()
            c_ranks = []
            c_doc_id = c_doc[0]['article_id']
            doc_ids_type = type(c_doc_id)
            if(doc_ids_type(ans_doc_id) == c_doc_id):
                print(len(has_ans_question), ' Found [', k, 'th] document', sep='')
                for c_sentence in c_doc:
                    c_rank = c_sentence['candidate_rank']
                    if(c_rank < max_c_num):
                        c_pos_range = range(c_sentence['answer_begin_position '], c_sentence['answer_end_position'])
                        if(ans_pos_range[0] in c_pos_range and ans_pos_range[1] in c_pos_range):
                            # print('full answer in sentence')
                            # print(ans_pos_range, c_pos_range)
                            # print(c_sentence['candidate_no'])
                            # print(c_sentence)
                            # print(c_rank)
                            c_ranks.append(c_rank)
                        elif(ans_pos_range[0] in c_pos_range and not(ans_pos_range[1] in c_pos_range)):
                            # print('part of answer in sentence')
                            # print(ans_pos_range, c_pos_range)
                            # print(c_sentence['candidate_no'])
                            c_ranks.append(c_rank)
                        elif(not(ans_pos_range[0] in c_pos_range) and ans_pos_range[1] in c_pos_range):
                            # print('part of answer in sentence')
                            # print(ans_pos_range, c_pos_range)
                            # print(c_sentence['candidate_no'])
                            c_ranks.append(c_rank)
                    # c_rank += 1
                if(not(c_ranks)):
                    ans_not_found_question.append(ans_doc_id)
                else:
                    has_ans_question.append(ans_doc_id)
                true_ans_ranks.append(c_ranks)
                break
            elif(k == len(candidate_answers[i])):
                print(doc_not_found, 'Document not found')
                true_ans_ranks.append([])
                doc_not_found_question.append(ans_doc_id)
                doc_not_found += 1
                break
            k += 1
    
    # candidate statistics
    has_answer_count = 0
    ans_not_found = -doc_not_found
    for candidate_ranks in true_ans_ranks:
        if(not(candidate_ranks)):
            ans_not_found += 1
        else:
            has_answer_count += 1

    print('Ranks of candidate sentences (Euclidean distance) that is true answer:\n', true_ans_ranks, sep='')
    print('Summary:\n\tHas answer:', has_answer_count, len(has_ans_question), 
        '\n\tAnswer not found:', ans_not_found, ans_not_found_question, 
        '\n\tDocument not found:', doc_not_found, doc_not_found_question)
    
    # histogram of ranks
    for i in range(len(true_ans_ranks)):
        if(true_ans_ranks[i]):
            for j in range(len(true_ans_ranks[i])):
                true_ans_ranks[i][j] += 1

    temp = []
    for ca in true_ans_ranks:
        if(ca):
            temp += ca
        else:
            # temp += [-2]
            pass
    true_ans_ranks = temp
    print(true_ans_ranks)

    bins = max_c_num
    plot_histogram('./results/reports/c_ranks1.png', true_ans_ranks, 
                    bins_num=bins, 
                    x_label='Candidates rank 1' + str(bins) + '', 
                    y_label='Occurrence')
    
    hansq = [2 for doc in has_ans_question]
    print(len(hansq))
    anfq = [1 for doc in ans_not_found_question]
    dnfq = [0 for doc in doc_not_found_question]
    has_ans_types = 'auto'
    candidate_stats = hansq + anfq + dnfq
    print(candidate_stats)

    plot_histogram('./results/reports/has_ans1.png', candidate_stats, 
                    bins_num=has_ans_types, 
                    x_label='Candidate answer statistics', 
                    y_label='Occurrence', 
                    legends=[True, ['0: document not found', '1: ans not found', '2: has ans'], 'best'])

if __name__ == "__main__":
    CANDIDATE_ANSWERS_PATH = './results/test_data/'
    candidate_answers_f = sorted(os.listdir(CANDIDATE_ANSWERS_PATH))
    print(candidate_answers_f)

    from time import sleep
    sleep(1)

    # get candidate ans from program
    candidate_answers = []
    for file_name in candidate_answers_f:
        with open(CANDIDATE_ANSWERS_PATH + file_name, 'r', encoding='utf-8', errors='ignore') as f:
            candidate_docs = json.load(f)
            for i in range(len(candidate_docs)):
                candidate_answers.append(candidate_docs[i])

    # get true answer details
    answer_details = {}
    with open('./data/new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        answer_details = json.load(f)
        answer_details = answer_details['data']
    
    # print(candidate_answers)
    # print(answer_details)
    
    check_origin(answer_details, candidate_answers)

    # a = []
    # for i in range(len(answer_details)):
    #     ans_doc_id = answer_details[i]['article_id']
    #     # print(ans_doc_id)
    #     a.append(ans_doc_id)
    
    # print(a[:20])