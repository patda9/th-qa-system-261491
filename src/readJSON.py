# coding=utf8

from pprint import pprint
import json
import os
# import deepcut
import time
# TODO fix this for ner algorithm

# s = 'นักบุญทาอีสในคริสต์ศตวรรษที่ 4เป็นชาวอะไร'
# s = 'ทางหลวงอินเตอร์สเตตอยู่บนเกาะอะไร'
# s = 'วอลเลซ เสต็กเนอร์เป็นใคร'
# s = 'ในอดีตเมืองคิตะกะมิเป็นส่วนหนึ่งของแคว้นอะไร' # 818026
# s = '"หนึ่งต่อเจ็ด" ออกฉายในปีใด' ##
# s = 'สมัยก่อนคิตะกะมิเป็นเมืองที่อยู่ในแคว้นใด' #
s = 'อุตะมะโระอาศัยอยู่ที่บ้านของใคร'

begin = time.time()
alphabet = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ'
vowel = 'เแโใไ'
e_alphabet = 'abcdefghijklmnopqrstuvwxyz'

index_dir = 'C:\\Users\\Patdanai\\Desktop\\dict2-20181109T145550Z-001\\dict2\\'

file = open("C:\\Users\\Patdanai\\Desktop\\drive-download-20181109T145340Z-001\\new_sample_questions_tokenize.json", mode = 'r' , encoding="utf-8-sig")
data = json.load(file)
validate = json.load(open("C:\\Users\\Patdanai\\Desktop\\drive-download-20181109T145340Z-001\\new_sample_questions_answer.json", mode = 'r' , encoding="utf-8-sig"))
# validate = [1]
# data = [deepcut.tokenize(s)]
doc = 3569
data = data[doc:4000]
print(data.__len__())
save = 1
string = ''
for s in data:
    string += "question " + str(doc)
    print("question",doc,s)

    start = time.time()

    # print(s)
    s.sort()
    # print(s)
    fs = []

    for w in s:
        fs.append(w[0])

    fs.sort()
    # print(fs)

    search = []
    cantfind = []
    notindex = []
    data = None
    for f in range(s.__len__()):
        if (s[f][0] != s[f-1][0]) and ((s[f][0] in alphabet) or (s[f][0] in e_alphabet) or (s[f][0] in vowel)) :
            file = open(index_dir + str(s[f][0]) + ".json")
            data = json.load(file)
        elif data == None:
            if (s[f] and (not s[f].isspace())):
                notindex.append(s[f])
            continue

        tmp = data.get(s[f])
        if  tmp != None :
            search.append((s[f], tmp))
        else:
            cantfind.append((s[f]))

    ## rank by (trial)tf-idf
    # rank = []
    # for i in search:
    #     rank.append([i[0],i[1][0][4]])

    # rank = sorted(rank,key=lambda l:l[1], reverse=True)
    # print(rank)

    intersection = []

    for i in range(search.__len__()) :
        intersection.append([])
        for j in search[i][1]:
            intersection[i].append(int(j[0]))

    intersection.sort(key=len)

    answer = set(intersection[0])
    for i in range(1,intersection.__len__()):
        if (set(intersection[i]) & answer):
            answer &= set(intersection[i])

    answer = list(answer)
    ans_int = ''
    # print(validate[doc])
    # print(cantfind)
    try:
        ans_int = ' ' + str(intersection[0].index(validate[doc])) + ' '
    except ValueError:
        ans_int = ' cant find in shortest '

    try:
        if answer.index(validate[doc]) < 6 :
            string += ': 1'
        else:
            string += ': 0'
        string += " rank" + str(answer.index(validate[doc])) + ' ||' + ans_int + str(cantfind) + str(notindex) + '\n'
    except ValueError:
        string += ": 0 Cant find doc" + ' ||' + ans_int + str(cantfind) + str(notindex) + '\n'

    end = time.time()
    print(end - start, 'secs')
    doc+=1
    save+=1
    if save==10 or doc==4000:
        with open("result_new.txt", "a" , encoding = "utf-8") as text_file:
            text_file.write(string)
        save = 0
        string = ''

end = time.time()
print(end - begin, 'secs')
# os.system("shutdown /s /t 90")
