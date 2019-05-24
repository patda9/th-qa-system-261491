# A Thai machine-learning based question-answering system from text source input project.

# Dataset description ([Google Drive](https://drive.google.com/drive/folders/1pPRewSKGxsbJSL4Zpfeydbormtf4vTml))
## Sentence Vectors Comparison Model training data
### data/compare_model_fasttext_dataset/
* positive_sentences.zip
> Description: Similar sentences based on 4,000 NSC 2018 questions use for question-answer sentences training.

> คำอธิบาย: ประโยคที่เหมือนซึ่งมาจาก 4,000 คำถามจาก NSC 2018 ใช้ในการสอนให้โมเดลเรียนรู้ความสัมพันธ์ของประโยคคำถามกับประโยคคำตอบ
* positive_embedded.zip
> Description: positive_sentences in fastText vector representation form.

> คำอธิบาย: ประโยคเดียวกันกับ positive_sentences แต่อยู่ในรูปเวกเตอร์จากโมเดล fastText
* negative0.zip
> Description: Dissimilar sentences based on 4,000 NSC 2018 questions and come from the same article with positive sentences use for question-answer sentences training.

> คำอธิบาย: ประโยคที่ไม่เหมือนซึ่งมาจาก 4,000 คำถามจาก NSC 2018 และอยู่ในบทความเดียวกับประโยคที่เหมือนใช้ในการสอนให้โมเดลเรียนรู้ความสัมพันธ์ของประโยคคำถามกับประโยคที่ไม่ใช่คำตอบ
* negative0_embedded.zip
> Description: negative0 in fastText vector representation form.

> คำอธิบาย: ประโยคเดียวกันกับ negative0 แต่อยู่ในรูปเวกเตอร์จากโมเดล fastText
* negative1.zip
> Description: Dissimilar sentences based on 4,000 NSC 2018 questions and come from the different article with positive sentences use for question-answer sentences training

> คำอธิบาย: ประโยคที่ไม่เหมือนซึ่งมาจาก 4,000 คำถามจาก NSC 2018 แต่ไม่ได้อยู่ภายในบทความเดียวกับประโยคที่เหมือนใช้ในการสอนให้โมเดลเรียนรู้ความสัมพันธ์ของประโยคคำถามกับประโยคที่ไม่ใช่คำตอบ
* negative1_embedded.zip
> Description: negative1 in fastText vector representation form.

> คำอธิบาย: ประโยคเดียวกันกับ negative1 แต่อยู่ในรูปเวกเตอร์จากโมเดล fastText


## Thai questions about articles in Wikipedia for question-answering
### data/nsc_2018_question/
* embedded_questions_4000_40_300.zip
> Description: fastText embedding questions, each question has equal length of 40 words and each word is embedded with 300 dimensional vector. (for question that has less than 40 words are padded with zero vector)

> คำอธิบาย: ไฟล์ประโยคที่แต่ละคำถูกเข้ารหัสด้วยโมเดล fastText ซึ่งแต่ละประโยคมีความยาวเท่ากันที่ 40 คำและแต่ละคำถูกแทนด้วยเวกเตอร์ขนาด 300 มิติ (ประโยคที่ยาวน้อยกว่า 40 คำจะถูกแทนด้วยเวกเตอร์ศูนย์)

* nsc_question_answers.json
> Description: This file provides question's answer details which consist of answer, answer beginning character position, answer endding character position, article ID, question and question ID in JSON object form.

> คำอธิบาย: ไฟล์เฉลยของคำถาม NSC 2018 จำนวน 4,000 คำถาม ประกอบด้วย คำตอบ, ตำแหน่งของตัวอักษรแรกของคำตอบ, ตำแหน่งของตัวอักษรสุดท้ายของคำตอบ, บทความของคำตอบ, คำถาม และเลขที่ของคำถามในรูปแบบ JSON object

Usage: 
```
import json
fp = open('nsc_question_answers.json', encoding='utf-8-sig')
data = json.load(fp)['data']
```
This will get list of 4,000 question answer objects.

Example: 

{

&nbsp;&nbsp;&nbsp;&nbsp;"question_id": 4,

&nbsp;&nbsp;&nbsp;&nbsp;"question": "กระทรวงโฆษณาแถลงข่าวและโฆษณาชวนเชื่อของนาซีเยอรมนี ก่อตั้งขึ้นในปี ค.ศ. ใด", 

&nbsp;&nbsp;&nbsp;&nbsp;"answer": "ปี 1933", 

&nbsp;&nbsp;&nbsp;&nbsp;"answer_begin_position ": 304, 

&nbsp;&nbsp;&nbsp;&nbsp;"answer_end_position": 311,

&nbsp;&nbsp;&nbsp;&nbsp;"article_id": 547560

}

* tokenized_questions.json
> Description: This file provides list of 4,000 tokenized questions using deepcut model.

> คำอธิบาย: ไฟล์คำถามที่ถูกตัดคำ (tokenize) แล้วซึ่งเป็นลิสต์ของคำถามจำนวน 4,000 คำถามที่ถูกตัดคำด้วยโมเดล deepcut

Usage: 
```
import json
fp = open('tokenized_questions.json', encoding='utf-8-sig')
data = json.load(fp)[3]
print(data)
```
**print(data)** will show

['กระทรวง', 'โฆษณา', 'แถลง', 'ข่าว', 'และ', 'โฆษณาชวนเชื่อ', 'ของ', 'นาซี', 'เยอรมนี', ' ', 'ก่อตั้ง', 'ขึ้น', 'ใน', 'ปี', ' ', 'ค.ศ.', ' ', 'ใด']

## fastText Thai pre-trained word vectors
### data/fasttext_th_wv/
> Description: This is word vectors for 2,000,000 Thai words trained from Thai Wikipedia.

> คำอธิบาย: ไฟล์เวกเตอร์ของคำที่ถูก Train ด้วยวิกิพีเดียภาษาไทยจำนวน 2,000,000 คำ

Usage:
```
fp = open('cc.th.300.vec', encoding='utf-8-sig')
vocab_wvs = {}
line_count = 0
for line in fp:
    if(line_count > 0):
        line = line.split()
        if(line[0] in vocabs):
            vocab_wvs[line[0]] = line[1:]
            vocab_count += 1
    line_count += 1
```

## Tokenized Thai Wikipedia articles using "deepcut" model
### data/tokenized_th_wiki/
