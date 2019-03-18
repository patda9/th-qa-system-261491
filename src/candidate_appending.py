import os
import json

PATH = './results/test_data/'
f_names = os.listdir(PATH)
print(len(f_names))

l = []
for i in range(len(f_names)):
    print(i)
    with open(PATH + f_names[i], encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    l.append(data)

with open('./results/test_data_4000/test_data_4000.json', 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(l, f, ensure_ascii=False)

with open('./results/test_data_4000/test_data_4000.json', encoding='utf-8', errors='ignore') as f:
    print(len(json.load(f)))
