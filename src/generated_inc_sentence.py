import json
import numpy as np
import os

# a = ['<doc id="', '540135', '"', ' ', 'url=', '"', 'https://th.', 'wikipedia.org/wiki?curid', '=', '540135', '"', ' ', 'title', '=', '"', 'เอมเมลี เด ฟอเรสต์', '"', '>', 'เอมเมลี เด ฟอเรสต์', ' เอมเมลี ชาร์ลอตต์', '-', 'วิกตอเรีย เด ฟอเร \
# สต์', ' ', '(', 'Emmelie', ' ', 'Charlotte', '-', 'Victoria de', ' ', 'Forest', ')', ' ', 'เกิด', 'วัน', 'ที่', ' ', '28', ' ', 'กุมภาพันธ์', ' ', 'ค.ศ.', ' ', '1993', ' ', 'เป็น', 'นัก', 'ร้อง', 'ชาว', 'เดนมาร์ก', ' ', 'เป็น', 'ตัว', 'แ\
# ทน', 'ของ', 'ประเทศเดนมาร์ก', 'ใน', 'การ', 'ประกวด', 'ยูโรวิชัน', ' ', '2013', ' ', 'ที่', 'เมืองมัลเมอ', ' ', 'ประเทศสวีเดน', ' ', 'โดย', 'เธอ', 'เป็น', 'ผู้', 'ชนะ', 'การ', 'แข่งขัน', 'ครั้ง', 'นี้', 'กับ', 'เพลง', ' ', '"', 'Only', '\
# ', 'Teardrops', '"', ' ', 'เธอ', 'เซ็น', 'สัญญา', 'กับ', 'ค่ายยูนิเวอร์ซัลมิวสิคกรุ๊ป', ' ', 'เมื่อ', 'วัน', 'ที่', ' ', '25', ' ', 'มีนาคม', ' ', 'ค.ศ.', ' ', '2013', ' ', 'มี', 'ผล', 'งาน', 'อัลบั้ม', 'แรก', 'ชื่อ', ' ', 'Only', ' ', '\
# Teardrops', ' ', 'ออก', 'วาง', 'ขาย', 'เดือน', 'พฤษภาคม', ' ', 'ค.ศ. 2013</doc>\n']

# tmp = ''
# for t in a:
#     tmp += t
# print(tmp.__len__())
# exit()

DOCUMENTS_PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'

with open('./data/new_sample_questions.json', encoding='utf-8') as f:
    answers_detail = json.load(f)['data']
    # print(answers_detail)

doc_f = os.listdir(DOCUMENTS_PATH)

word_per_sentence = 20

k = 0
incorrect_sentences = []
for doc in answers_detail:
    with open(DOCUMENTS_PATH + str(doc['article_id']) + '.json', encoding='utf-8') as f:
        abp = doc['answer_begin_position '] - 1
        aep = doc['answer_end_position']
        tokens = json.load(f)
        
        cumulative_len = 0
        index = 0
        
        for token in tokens:
            if(cumulative_len in range(abp, aep)):
                answer_index = index
            cumulative_len += len(token)
            index += 1
        
        # print(answer_index, cumulative_len, list(range(abp, aep)), doc['question_id'])
        sentences = []
        for i in range(0, answer_index-word_per_sentence, word_per_sentence):
            sentence = tokens[i:i+word_per_sentence]
            # print(answer_index, i, '*', sentence)
            if(sentence):
                sentences.append(sentence)
            else:
                sentences.append(tokens[0:word_per_sentence])
                while(len(sentence) < word_per_sentence):
                    sentence.insert(0, '<PAD>')
        if(not(sentences)):
            sentences.append(tokens[0:word_per_sentence])
        for s in sentences:
            while(len(s) < word_per_sentence):
                s.insert(0, '<PAD>')
    incorrect_sentences.append(sentences)
    k += 1

with open('./results/incorrect_sentences_tokens/additional_incorrect_sentences.json', 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(incorrect_sentences, f)

with open('./results/incorrect_sentences_tokens/additional_incorrect_sentences_readable.json', 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(incorrect_sentences, f, ensure_ascii=False, indent=4)
