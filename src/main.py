import findDocuments as fd
import findAnswer as fa
import json
import numpy as np
import preprocessing as prep
import os
import re
import sentence_vectorization as sv
import candidates_listing as cl
import candidates_sentences_selection as css

from gensim.models import Word2Vec
from keras.models import load_model, Model
from pprint import pprint

if(__name__ == '__main__'):
    # candidate_output = fd.findDocuments(end_idx=None)
    # with open('./results/final/candidate_doc_ids.json', 'w', errors='ignore') as find_doc:
    #     json.dump(candidate_output, find_doc)
    
    # candidate_document_ids = fd.findDocuments()
    # candidate_sentences = css.candidate_similarity()
    # css.candidate_similarity(candidate_document_ids)
    candidate_sentences = []
    PATH = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/results/final/candidate_sentences/'
    files = os.listdir(PATH)
    for i in range(len(files)):
        with open(PATH + files[i], encoding='utf-8') as f:
            temp = json.load(f)
        candidate_sentences.append(temp)
        print('File: %d was read' % (i))
    doc_n = 7
    fa.find_answer(candidate_sentences)
    