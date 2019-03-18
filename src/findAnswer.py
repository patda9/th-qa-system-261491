import json
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)


def normalized_edit_similarity(a, b):
    import editdistance
    return 1.0 - editdistance.eval(a, b) / (1.0 * max(len(a), len(b)))


def similar(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def extractNumberFromString(string):
    import re
    return re.findall('\d+', string)


def hasNumbers(inputString, thai_number_text):
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
    return [{'sentence': sentence_answer, 'begin_position': l}]


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


def find_candidate(possible_answer, doc_id, sentence_answer, l, word_class,sp):
    temp = []
    for w in sentence_answer:
        if(not(w) in sp):
            temp.append(w)
    
    sentence_answer = temp
    # print(sentence_answer)
    for n in range(2):
        for k in range(sentence_answer.__len__()):
            # print(type(sentence_answer[k]), sentence_answer[k])
        
            if n == 1 and possible_answer.__len__() < 1:
                possible_answer.append(sentence_answer[k])
                doc_id.append(k)
            elif sentence_answer[k] not in sp and not sentence_answer[k].isnumeric():
                if sentence_answer[k] in word_class[l] :
                    possible_answer.append(sentence_answer[k])
                    doc_id.append(k)
    return possible_answer, doc_id


def relevance_score(question, sentence, similarity_score,max_similarity_score, doc_rank,doc_n, candidate, question_word):
    a = []
    question_word_index = question.index(question_word)
    l = 2 * question.__len__()
    for i in candidate:
        a.append([])
        for j in range(i - l, i + l):
            if (i != j) and (0 <= j < sentence.__len__()) and (sentence[j] in question):
                if question.index(sentence[j]) < question_word_index:
                    a[-1].append([question.index(sentence[j]), j, 1])
                else:
                    a[-1].append([question.index(sentence[j]), j, 1])
        # print(a[-1])

    m = question.__len__() - 1

    score = []
    for i in range(a.__len__()):
        tmp = 0
        for j in a[i]:
            tmp += (1 - abs(j[1] - candidate[i]) / l) * (1 - abs(j[0] - question_word_index) / m) * (1 - doc_rank/doc_n) #* (1 - similarity_score/max_similarity_score)
        score.append(tmp)

    return score


def find_answer_word(question,j,max_similarity_score, rand, doc_id, question_word_index, answer_position,doc_n=7):
    # print(possible_answer[-1])
    # print(question_word_index)
    # print(j['sentence'], doc_id[-1])

    score = relevance_score(question, j['sentence'], j['similarity_score'],max_similarity_score, j['doc_rank'],doc_n, doc_id[-1][-1][1:], question_word_index[1])

    tmp = doc_id[-1][-1][score.index(max(score)) + 1]
    doc_id[-1][-1].insert(1, j['sentence'][tmp])
    for word in j['sentence']:
        if word != j['sentence'][tmp]:
            rand += word.__len__()
        else:
            break
    answer_position[-1].append([rand, rand + j['sentence'][tmp].__len__()])

    return max(score)


def findAnswer(question, inp):
    doc_id = []
    question_type = [
        ['กี่', 'ปี ใด', 'ปี อะไร', 'พ.ศ.  อะไร', 'ค.ศ.  อะไร', 'พ.ศ. อะไร', 'ค.ศ. อะไร', 'พ.ศ. ใด', 'พ.ศ.  ใด',
         'ค.ศ. ใด',
         'ค.ศ.  ใด', 'เท่า ไร', 'เท่า ไหร่', 'เท่า ใด', 'คริสต์ศักราช ใด', 'จำนวน ใด']
        , ['เมื่อ ไร', 'เวลา ใด', 'วัน ใด', 'เมื่อ ใด', 'วัน ที่']  # date format
        , ['ใคร', 'ว่า อะไร', 'ชื่อ อะไร', 'คน ใด', 'คน ไหน', 'คือใคร', 'ผู้ ใด']  # human name
        , ['ประเทศ ใด', 'ประเทศ อะไร'
            , 'จังหวัดใด', 'จังหวัด ใด', 'จังหวัด อะไร'
            , 'เมืองใด', 'เมือง ใด', 'เมือง อะไร'
            , 'ภาค ใด'
            , 'แคว้น ใด'
            , 'ทวีปใด', 'ทวีป อะไร', 'ทวีป ใด', 'ภูมิภาค ไหน'
            , 'ที่ ไหน', 'ที่ ใด', 'ใด', 'ไหน']  # where
        , ['อะไร', 'อย่าง ไร']  # other what, other dai, other nhai
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
             'พ.ศ.', 'ค.ศ.']
    special_char = "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "
    class_label = [2, 3, 4]
    word_class = [[], []]
    for i in class_label:
        tmp = json.load(open("./word_10class/" + str(i) + ".json", "r", encoding="utf-8"))
        word_class.append(set(tmp))

    wrong = 0
    possible_answer = []
    answer_position = []

    for_answer_json = {}
    for_answer_json['data'] = []
    print(inp.__len__())
    for i in range(wrong, inp.__len__()):
        s = ''.join(question[i])
        try:
            max_similarity_score = inp[i][-1]['similarity_score']
        except:
            max_similarity_score = -1

        possible_answer.append([])
        doc_id.append([])
        answer_position.append([])
        print(i+1,"/",inp.__len__(), s)
        rr_score = []
        for l in range(question_type.__len__()):
            if any(check_question_type(k, question[i]) for k in question_type[l]):
                question_word_index = find_question_word(question[i], question_type[l])
                for j in inp[i]:
                    doc_id[-1].append([j["article_id"]])
                    possible_answer[-1].append([])
                    if l > 1:
                        possible_answer[-1][-1], doc_id[-1][-1] = find_candidate(possible_answer[-1][-1],
                                                                                 doc_id[-1][-1],
                                                                                 j['sentence'], l, word_class,special_char)
                        rr_score.append(find_answer_word(question[i],j,max_similarity_score, j['answer_begin_position '], doc_id, question_word_index,
                                                         answer_position))
                    elif l == 0:

                        for k in range(j['sentence'].__len__()):
                            if hasNumbers(j['sentence'][k], thai_number_text):
                                doc_id[-1][-1].append(k)
                                possible_answer[-1][-1].append(j['sentence'][k])
                        if possible_answer[-1][-1].__len__() < 1:
                            for k in range(j['sentence'].__len__()):
                                doc_id[-1][-1].append(k)
                                possible_answer[-1][-1].append(j['sentence'][k])
                        rr_score.append(find_answer_word(question[i],j,max_similarity_score, j['answer_begin_position '], doc_id, question_word_index,
                                                         answer_position))
                    else:
                        for k in range(j['sentence'].__len__()):
                            if j['sentence'][k] in month:
                                doc_id[-1][-1].append(k)
                                possible_answer[-1][-1].append(j['sentence'][k])
                            else:
                                possible_answer[-1][-1], doc_id[-1][-1] = find_candidate(possible_answer[-1][-1],
                                                                                         doc_id[-1][-1], j['sentence'],
                                                                                         l, word_class,special_char)
                        rr_score.append(find_answer_word(question[i],j,max_similarity_score, j['answer_begin_position '], doc_id, question_word_index,
                                                         answer_position))
                break

            elif l == class_label[-1] and not any(check_question_type(k, question[i]) for k in question_type[l]):
                tmp_q = []
                for q in question[i]:
                    tmp = []
                    for w in question_type[l]:
                        tmp.append(similar(q, w))
                    tmp_q.append([question[i].index(q), max(tmp)]) ### 
                tmp_q.sort(key=lambda s: s[1], reverse=True)
                question_word_index = [tmp_q[0][0], question[i][tmp_q[0][0]]]
                for j in inp[i]:
                    doc_id[-1].append([j["article_id"]])
                    possible_answer[-1].append([])
                    # print("\n#############################\n")
                    possible_answer[-1][-1], doc_id[-1][-1] = find_candidate(possible_answer[-1][-1], doc_id[-1][-1],
                                                                             j['sentence'], l, word_class,special_char)
                    rr_score.append(
                        find_answer_word(question[i],j,max_similarity_score, j['answer_begin_position '], doc_id, question_word_index, answer_position))

        # print(rr_score)
        # print(rr_score.index(max(rr_score)), doc_id[i][rr_score.index(max(rr_score))])
        # print(''.join(inp[i][rr_score.index(max(rr_score))]['sentence']))
        
        try:
            temp = {
                'question_id': i + 1, 
                'question': s, 
                'answer': doc_id[i][rr_score.index(max(rr_score))][1], 
                'answer_begin_position ': answer_position[i][rr_score.index(max(rr_score))][0], 
                'answer_end_position': answer_position[i][rr_score.index(max(rr_score))][1], 
                'article_id': doc_id[i][rr_score.index(max(rr_score))][0]
            }
            
        except:
            temp = {
            'question_id': i + 1, 
            'question': s, 
            'answer': "EXCEPT", 
            'answer_begin_position ': 0, 
            'answer_end_position': 0, 
            'article_id': -1
            }
        for_answer_json['data'].append(temp)
        
        
    return for_answer_json

def restruct_candidate_sentences(sentences):
    saved = []
    for i in range(sentences.__len__()):
        for j in range(sentences[i].__len__()):
            print(i * sentences[i].__len__() + j)
            saved.append([])
            for k in range(sentences[i][j].__len__()):
                sentences[i][j][k] = sorted(sentences[i][j][k], key=lambda s: s['similarity_score'],reverse=True)
                for l in sentences[i][j][k][:10]:
                    l['doc_rank'] = k
                    del l["candidate_rank"]
                    del l["answer_end_position"]
                    saved[-1].append(l)
    for i in range(saved.__len__()):
        print(str(i + 1) + "/" + str(saved.__len__()))

        saved[i] = saved[i]
    return saved

def correct_sentences():
    inp = json.load(open("E:\CPE#Y4\databaseTF\\new_model_dataset\\correct.json", "r", encoding="utf-8"))
    tmp = []
    n = 0
    for i in inp:

        if i["question_id"] == n:
            i['doc_rank'] = 0
            i["answer_begin_position "] = 0
            i['similarity_score'] = 0
            i['sentence'] = i["sentence_tokens"]
            tmp.append([i])
            n += 1
    return tmp

def find_answer(inp):
    #  q = open('./data/final/final_tokenized_question.json', mode='r', encoding="utf-8-sig") # change path
    # question = json.load(open('ThaiQACorpus-EvaluationDataset-tokenize.json', 'r', encoding='utf-8-sig'))
    question = json.load(open('./data/final/final_tokenized_question.json', mode='r', encoding="utf-8-sig"))
    # inp = json.load(open("E:\CPE#Y4\databaseTF\candidate_sentence\\candidate_output_4000x7.json", "r", encoding="utf-8"))
    print(question.__len__())

    tmp = []
    # for i in question:
    #     tmp.append(i['sentence'])
    #     for j in range(tmp[-1].__len__()):
    #         if '<PAD>' in tmp[-1]:
    #             tmp[-1].remove('<PAD>')
    #         else:
    #             break
    # question = tmp
    inp = restruct_candidate_sentences(inp)
    doc_n = 7

    answer_json = findAnswer(question, inp)
    with open('./output/21p31n0105_eval_5000.json', 'w', encoding="utf-8") as outfile:
        json.dump(answer_json, outfile, ensure_ascii=False)
