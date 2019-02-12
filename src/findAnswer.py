import json
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
from gensim.models import Word2Vec
import numpy as np
from pythainlp.number import *


def normalized_edit_similarity(a, b):
    import editdistance
    return 1.0 - editdistance.eval(a, b) / (1.0 * max(len(a), len(b)))


def similar(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def extractNumberFromString(string):
    import re
    return re.findall('\d+', string)


def hasNumbers(inputString):
    for i in thai_number_text:
        if inputString.startswith(i) or inputString.endswith(i):
            return True
    return any(char.isdigit() for char in inputString)


def make_sentence_answer(article_id, answer_begin, n=15):
    doc = json.load(
        open('E:\CPE#Y4\databaseTF\documents-tokenize\\' + str(article_id) + '.json', 'r',
             encoding='utf-8-sig'))
    sentence_answer = []
    l = 0
    for j in range(doc.__len__()):
        l += doc[j].__len__()
        if l >= answer_begin - 1:
            for k in range(j - n, j + n):
                if k < j:
                    l -= doc[k].__len__()
                try:
                    sentence_answer.append(doc[k])

                except IndexError:
                    break
            break
    return [{'sentence':sentence_answer,'begin_position':l}]


def check_question_type(a, question):
    c = a.split()
    n = a.count(' ')

    if n == 0:
        return c[0] in question
    else:
        return c[1] in question and question[question.index(c[1]) - n] == c[0]


def find_question_word(question, question_words):
    for w in question_words:
        c = w.split()
        n = w.count(' ')
        if check_question_type(w, question):
            if n == 0:
                return [question.index(c[0]), c[0]]
            else:
                return [question.index(c[1]), c[1]]


def find_candidate(possible_answer, doc_id, sentence_answer, l ,word_class):
    n=0
    while possible_answer.__len__() < 1 :
        for k in range(sentence_answer.__len__()):
            if sentence_answer[k] != ' ' and not sentence_answer[k].isnumeric():
                if sentence_answer[k] in word_class[l+n]:
                    possible_answer.append(sentence_answer[k])
                    doc_id.append(k)
        n+=1

    return possible_answer, doc_id
## TODO edit find candidate

def relevance_score(question, sentence, candidate, question_word):
    a = []
    question_word_index = question.index(question_word)
    l = 2 * question.__len__()
    for i in candidate:
        a.append([])
        for j in range(i - l, i + l):
            if (i != j) and (0 <= j < sentence.__len__()) and (sentence[j] in question):
                if question.index(sentence[j]) < question_word_index:
                    a[-1].append([question.index(sentence[j]), j, 0.5])
                else:
                    a[-1].append([question.index(sentence[j]), j, 0.25])
        # print(a[-1])

    m = question.__len__() - 1

    score = []
    for i in range(a.__len__()):
        tmp = 0
        for j in a[i]:
            tmp += (1 - abs(j[1] - candidate[i]) / l) * (1 - abs(j[0] - question_word_index) / m)
        score.append(tmp)

    return score


def find_answer_word(j,rand):
    print(i, s)
    print(possible_answer[-1])
    print(question_word_index, question[i])
    print(j['sentence'], doc_id[-1])

    score = relevance_score(question[i], j['sentence'], doc_id[-1][-1][1:], question_word_index[1])

    tmp = doc_id[-1][-1][score.index(max(score)) + 1]
    doc_id[-1][-1].insert(1, j['sentence'][tmp])
    for word in j['sentence']:
        if word != j['sentence'][tmp]:
            rand += word.__len__()
        else:
            break
    answer_position[-1].append([rand, rand + j['sentence'][tmp].__len__()])

    return max(score)

if(__name__ == '__main__'):
    a = json.load(open('test_set/new_sample_questions.json', encoding='utf-8-sig'))
    a = a['data']
    question = json.load(open('test_set\\new_sample_questions_tokenize.json', 'r', encoding='utf-8-sig'))

    question_index = []
    doc_id = []
    real_answer = []
    question_type = [
        ['กี่', 'ปี ใด', 'ปี อะไร', 'พ.ศ.  อะไร', 'ค.ศ.  อะไร', 'พ.ศ. อะไร', 'ค.ศ. อะไร', 'พ.ศ. ใด', 'พ.ศ.  ใด', 'ค.ศ. ใด',
        'ค.ศ.  ใด', 'เท่า ไร', 'เท่า ไหร่', 'เท่า ใด', 'คริสต์ศักราช ใด', 'จำนวน ใด']
        , ['เมื่อ ไร', 'เวลา ใด', 'วัน ใด', 'เมื่อ ใด', 'วัน ที่']  # date format
        , ['ใคร', 'ว่า อะไร', 'ชื่อ อะไร', 'คน ใด', 'คน ไหน', 'คือใคร', 'ผู้ ใด']  # human name
        , ['ประเทศ ใด', 'ประเทศ อะไร']
        , ['จังหวัดใด', 'จังหวัด ใด', 'จังหวัด อะไร']
        , ['เมืองใด', 'เมือง ใด', 'เมือง อะไร']
        , ['ภาค ใด']
        , ['แคว้น ใด']
        , ['ทวีปใด', 'ทวีป อะไร', 'ทวีป ใด', 'ภูมิภาค ไหน']
        , ['ที่ ไหน', 'ที่ ใด']  # where
        , ['อะไร', 'อย่าง ไร', 'ใด', 'ไหน']  # other what, other dai, other nhai
    ]
    thai_number_text = [u'หนึ่ง', u'สอง', u'สาม', u'สี่', u'ห้า', u'หก', u'เจ็ด', u'แปด', u'เก้า', u'สิบ', u'สิบเอ็ด']
    month = ['มกราคม', 'ม.ค.',
            'กุมภาพันธ์', 'ก.พ.',
            'มีนาคม', 'มี.ค.',
            'เมษายน', 'เม.ย.',
            'พฤษภาคม', 'พ.ค.',
            'มิถุนายน', 'มิ.ย.',
            'กรกฎาคม', 'ก.ค.',
            'สิงหาคม', 'ส.ค.',
            'กันยายน', 'ก.ย.',
            'ตุลาคม', 'ต.ค.',
            'พฤศจิกายน', 'พ.ย.',
            ' ธันวาคม', 'ธ.ค.',
            'พ.ศ.','ค.ศ.']

    class_label = [2,3,4,5,6,7,8,9,10]
    word_class = [[],[]]
    for i in class_label:
        tmp = json.load(open("./data/word_class/" + str(i) + ".json", "r", encoding="utf-8"))
        word_class.append(set(tmp))
    wrong = 0
    all_rs = []
    possible_answer = []
    answer_position = []
    answer_json = []

    inp = json.load(open("candidates_out.json", "r", encoding="utf-8"))
    for i in range(inp.__len__()):
        inp[i] = inp[i][0]

    for i in range(wrong, inp.__len__()):
        answer = a[i]['answer']
        # inp.append(make_sentence_answer(article_id, answer_begin))  ### input
        real_answer.append(answer)
        s = ''.join(question[i])
        possible_answer.append([])
        doc_id.append([])
        answer_position.append([])

        rr_score = []
        for l in range(question_type.__len__()):
            if any(check_question_type(k, question[i]) for k in question_type[l]):
                question_word_index = find_question_word(question[i], question_type[l])
                for j in inp[i]:
                    doc_id[-1].append([j["article_id"]])
                    possible_answer[-1].append([])
                    if l > 1:
                        possible_answer[-1][-1], doc_id[-1][-1] = find_candidate(possible_answer[-1][-1], doc_id[-1][-1], j['sentence'], l, word_class)
                        rr_score.append(find_answer_word(j,j['answer_begin_position ']))
                    elif l == 0:
                        for k in range(j['sentence'].__len__()):
                            if hasNumbers(j['sentence'][k]):
                                doc_id[-1][-1].append(k)
                                possible_answer[-1][-1].append(j['sentence'][k])
                        if possible_answer[-1][-1].__len__() < 1 :
                            possible_answer[-1].pop()
                            doc_id[-1].pop()
                            continue
                        rr_score.append(find_answer_word(j,j['answer_begin_position ']))
                    else:
                        for k in range(j['sentence'].__len__()):
                            if j['sentence'][k] in month:
                                doc_id[-1][-1].append(k)
                                possible_answer[-1][-1].append(j['sentence'][k])
                            else:
                                possible_answer[-1][-1], doc_id[-1][-1] = find_candidate(possible_answer[-1][-1], doc_id[-1][-1],j['sentence'], l)
                        rr_score.append(find_answer_word(j,j['answer_begin_position ']))
                break

            elif l == 10 and not any(check_question_type(k, question[i]) for k in question_type[l]):
                tmp_q = []
                for q in question[i]:
                    tmp = []
                    for w in question_type[l]:
                        tmp.append(similar(q, w))
                    tmp_q.append([question[i].index(q), max(tmp)])
                tmp_q.sort(key=lambda s: s[1], reverse=True)
                question_word_index = [tmp_q[0][0], question[i][tmp_q[0][0]]]
                for j in inp[i]:
                    doc_id[-1].append([j["article_id"]])
                    possible_answer[-1].append([])
                    print("\n#############################\n")
                    possible_answer[-1][-1], doc_id[-1][-1] = find_candidate(possible_answer[-1][-1], doc_id[-1][-1], j['sentence'], l)
                    rr_score.append(find_answer_word(j,j['answer_begin_position ']))

        print(doc_id[i][rr_score.index(max(rr_score))])

        for_answer_json = {}
        for_answer_json['question_id'] = i+1
        for_answer_json['question'] = s
        for_answer_json['answer'] = doc_id[i][rr_score.index(max(rr_score))][1]
        for_answer_json['answer_begin_position '] = answer_position[i][rr_score.index(max(rr_score))][0] #### w8 input
        for_answer_json['answer_end_position'] = answer_position[i][rr_score.index(max(rr_score))][1] #### w8 input
        for_answer_json['article_id'] = doc_id[i][rr_score.index(max(rr_score))][1] #### w8 input
        answer_json.append(for_answer_json)

    with open('output_answer.json', 'w' , encoding="utf-8") as outfile:
        json.dump(answer_json,outfile,indent=4,ensure_ascii=False)
