import json

with open('./../new_sample_questions_tokenize.json', 'r', encoding='utf-8') as f1:
    questions = json.load(f1)

with open('./../new_sample_questions_answer.json', 'r', encoding='utf-8') as f2:
    answer_doc_id = json.load(f2)

# q = [w for w in questions if w.strip()]

for i in range(len(questions)):
    questions[i] = [w for w in questions[i] if not(w is ' ')]
    print(questions[i])
