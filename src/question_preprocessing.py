import json
import numpy

questions_path = 'C:/Users/Patdanai/Workspace/th-qa-system-261491/data/final/ThaiQACorpus-EvaluationDataset_Final_QuestionList.json'
word_per_sentence = 40

with open(questions_path, 'r', encoding='utf-8', errors='ignore') as f:
    nsc_questions = json.load(f)['data']

q_tokens = []
with open('./data/final/final_tokenized_question.json', 'r', encoding='utf-8', errors='ignore') as f:
    questions = json.load(f)

print(questions.__len__())
# print(nsc_answer_details.__len__())

for i in range(len(questions)):
    while(len(questions[i]) < word_per_sentence):
        questions[i].insert(0, '<PAD>')
    while(len(questions[i]) > word_per_sentence):
        questions[i] = questions[i][len(questions[i]) - word_per_sentence:]

print(questions)

questions_tokens = []
for i in range(len(nsc_questions)):
    questions_tokens.append({
            # "article_id": nsc_questions[i]['article_id'], 
            "question_id": i, 
            "sentence_tokens": questions[i]
        })

print(questions_tokens[0])
print(len(questions_tokens))

with open('./results/final/question_sentence_tokens.json', 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(questions_tokens, f, indent=4)

with open('./results/final/question_sentence_tokens_readable.json', 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(questions_tokens, f, ensure_ascii=False, indent=4)
