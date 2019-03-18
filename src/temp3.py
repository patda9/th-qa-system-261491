# from deepcut import tokenize
import json

# data = []
# with open('C:/Users/Patdanai/Workspace/th-qa-system-261491/data/final/ThaiQACorpus-EvaluationDataset_Final_QuestionList.json', encoding='utf-8', errors='ignore') as f:
#     data = json.load(f)['data']

with open('./results/final/candidate_doc_ids.json', encoding='utf-8') as f:
    data = json.load(f)
    print(data[862:865])

# tokenized_questions = []
# for i in range(len(data)):
#     print(i)
#     tokenized_questions.append(tokenize(data[i]['question']))
    # print(tokenized_questions)

# with open('./data/final/final_tokenized_question.json', 'w', encoding='utf-8', errors='ignore') as f:
    # json.dump(tokenized_questions, f)


