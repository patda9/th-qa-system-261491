import deepcut
import json
#import allah
file = open("./data/qa-output100.json", encoding="utf-8-sig")
data = json.load(file)

# for k in data :
#     print(k)

data = data['data']
# print(data)

answers = []
questions = []

# print(data[0])
for i in data:
    questions.append(i['question'])
    # print(i['question'])
    answers.append(i['answer'])
    # print(i['answer'])

# print(questions)
# print(answers)

segmented_q = []
unk_q = []
question_identifiers = ['ไหน', 'ใด', 'อะไร', 'ไร', 'ใคร', 'ไหร่', 'กี่', '<UNK>']
n = [0] * len(question_identifiers)

for i, q in enumerate(questions):
    segmented_q.append(deepcut.tokenize(q))
    unk_q.append(1)

for i in range(len(question_identifiers)):
    print(question_identifiers[i])
    for j in range(len(segmented_q)):
        if((question_identifiers[i] in segmented_q[j])):
            print(j, segmented_q[j], question_identifiers[i])
            unk_q[j] = 0
            n[i] += 1
    print()

for i in range(len(unk_q)):
    if(unk_q[i] == 1):
        print(i, segmented_q[i], question_identifiers[7])
print()

n[7] = sum(unk_q)

print(question_identifiers)
print(n)

