import deepcut
import json
import numpy as np

from positive_generator import answers_detail, get_vocab_wvs, preprocess_document, vectorize_tokens

path = './data/nsc_questions_answers.json'
wv_path = 'C:/Users/Patdanai/Workspace/cc.th.300.vec'
words_per_sample = 40

def load_data(data=None, path=None):
    if(path):
        with open(path, encoding='utf-8-sig') as f:
            data = json.load(f)

    return data

def tokenize(sentence):
    tkned_s = deepcut.tokenize(preprocess_document(sentence))
    return tkned_s

if __name__ == "__main__":
    questions = []
    for q in answers_detail:
        questions.append(q['question'])

    for i in range(len(questions)):
        questions[i] = tokenize(questions[i])
        while(len(questions[i]) < words_per_sample):
            questions[i].insert(0, '<PAD>')

    vocabs = set([tk for q in questions for tk in q if not tk in ['', ' ']])

    vocab_wvs = get_vocab_wvs(wv_path, vocabs=vocabs)

    eq = []
    for q in questions:
        wvs = vectorize_tokens(q, vocab_wvs=vocab_wvs)
        eq.append(wvs)

    out_file_name = './data/dataset/questions/tokenized_questions.json'
    with open(out_file_name, 'w', encoding='utf-8-sig') as f:
        json.dump(questions, f, ensure_ascii=0)

    out_file_name_npy = './data/dataset/questions/embedded_questions_4000_40_300.npy'
    np.save(out_file_name_npy, np.array(eq))
