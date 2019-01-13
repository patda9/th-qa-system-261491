import json
import re

# defining regex pattern
pattern = re.compile(r"[^\u0E00-\u0E7F0-9]|^'|'$|''")

def load_article(dir_path, art_id):
    with open(dir_path + art_id + '.json', 'r', encoding='utf-8', errors='ignore') as f:
        article = json.load(f)
    return article

# remove xml tag at the beginning
def remove_xml(article):
    xml_close_index = 0
    for i in range(len(article)):
        if(article[i] == '>'):
            xml_close_index = i
    return article[xml_close_index:]

# remove english, special char, *(and stop words)  
def remove_stop_words(article, pattern=re.compile(r"[^\u0E00-\u0E7F0-9]|^'|'$|''")):
    for i in range(len(article)):
        char_to_remove = re.findall(pattern, article[i])
        list_with_char_removed = [char for char in article[i] if(not(char in char_to_remove))]
        article[i] = ''.join(list_with_char_removed)
    return list(filter(None, article))