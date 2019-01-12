import json

with open('./new_sample_questions_tokenize.json', 'r', encoding='utf-8') as f1:
    questions = json.load(f1)

with open('./new_sample_questions_answer.json', 'r', encoding='utf-8') as f2:
    answer_doc_id = json.load(f2)

for i in range(len(questions)):
    print(questions[i], answer_doc_id[i])
    print()
    