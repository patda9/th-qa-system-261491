# -*- coding: utf-8 -*-

import deepcut
""" 
tokenizing wiki-corpus 
input: text_files
output: <list>words, <list> words_id
How to be bald
"""
with open('C:\\Users\\Patdanai\\Desktop\\th-qa-system-261491\\documents-nsc\\665.txt', encoding='utf-8') as f:
    for l in f:
        source = l

a = deepcut.tokenize(source)
a = [a[i * 20:(i + 1) * 20] for i in range((len(a) + 20 - 1) // 20 )]  
print(a)

