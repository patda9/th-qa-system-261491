# th-qa-system-261491
A Thai machine-learning based question-answering system from text source input project.

# Dataset description
## Sentence Vectors Comparison Model training data
### data/compare_model_fasttext_dataset/
## Thai questions about articles in Wikipedia for question-answering
### data/nsc_2018_question/
* embedded_questions_4000_40_300.zip
> Description: fastText embedding questions, each question has equal length of 40 words and each word is embedded with 300 dimensional vector. (for question that has less than 40 words are padded with zero vector)
คำอธิบาย: ไฟล์ประโยคที่แต่ละคำถูกเข้ารหัสด้วยโมเดล fastText ซึ่งแต่ละประโยคมีความยาวเท่ากันที่ 40 คำและแต่ละคำถูกแทนด้วยเวกเตอร์ขนาด 300 มิติ (ประโยคที่ยาวน้อยกว่า 40 คำจะถูกแทนด้วยเวกเตอร์ศูนย์)
* nsc_question_answers.json
Usage: 
```
import json
fp = open('nsc_question_answers.json', encoding='utf-8-sig')
data = json.load(fp)['data']
```
this will get list of 4,000 question answer objects.
> Description: This file provides question's answer details which consist of answer, answer beginning character position, answer endding character position, article ID, question and question ID in JSON object form
คำอธิบาย: 

> Example: 
{
&nbsp;&nbsp;&nbsp;&nbsp;"question_id": 4,
&nbsp;&nbsp;&nbsp;&nbsp;"question": "กระทรวงโฆษณาแถลงข่าวและโฆษณาชวนเชื่อของนาซีเยอรมนี ก่อตั้งขึ้นในปี ค.ศ. ใด", 
&nbsp;&nbsp;&nbsp;&nbsp;"answer":"ปี 1933", 
&nbsp;&nbsp;&nbsp;&nbsp;"answer_begin_position ": 304, 
&nbsp;&nbsp;&nbsp;&nbsp;"answer_end_position": 311,
&nbsp;&nbsp;&nbsp;&nbsp;"article_id": 547560
}

* tokenized_questions.json
> Description: This file provides list of tokenized questions using deepcut model
คำอธิบาย: 

> Example: 
```
import json
fp = open('tokenized_questions.json', encoding='utf-8-sig')
data = json.load(fp)[3]
print(data)
```
"print(data)" will show ['กระทรวง', 'โฆษณา', 'แถลง', 'ข่าว', 'และ', 'โฆษณาชวนเชื่อ', 'ของ', 'นาซี', 'เยอรมนี', ' ', 'ก่อตั้ง', 'ขึ้น', 'ใน', 'ปี', ' ', 'ค.ศ.', ' ', 'ใด']
## fastText Thai pre-trained word vectors
### data/fasttext_th_wv/
## Tokenized Thai Wikipedia articles using "deepcut" model
### data/tokenized_th_wiki/
