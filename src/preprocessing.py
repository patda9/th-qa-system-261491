import json
import re

# defining regex pattern
# pattern = re.compile(r"[^\u0E00-\u0E7F0-9]|^'|'$|''")
path = 'C:/Users/Patdanai/Desktop/wiki-dictionary-[1-50000]/' # default development path

def load_article(dir_path, art_id):
    with open(dir_path + str(art_id) + '.json', 'r', encoding='utf-8', errors='ignore') as f:
        article = json.load(f)
    return article

# remove xml tag at the beginning
def remove_xml(article):
    xml_close_index = 0
    for i in range(len(article)):
        if(article[i] == '>'):
            xml_close_index = i + 1
    return article[xml_close_index:]

# remove english, special char, *(and stop words) returns remaining words from preprocessing and origin cumulative string lengths
def remove_noise(article, pattern=re.compile(r"[^\u0E00-\u0E7F^0-9^ \t^.^,]")):
    list_with_char_removed = []
    word_lengths = []
    for i in range(article.__len__()):
        char_to_remove = re.findall(pattern, article[i])
        temp = ''
        word_lengths.append(article[i].__len__())
        for j in range(article[i].__len__()):
            if(not(article[i][j] in char_to_remove)):
                temp += article[i][j]
        if(temp.__len__() and not(temp in re.findall(r"^\s|^,|^.", temp))):
            list_with_char_removed.append((i, temp))
    word_locations = word_lengths.copy()
    summation = 0
    for i in range(word_locations.__len__()):
        summation += word_locations[i]
        word_locations[i] = summation
    # print(word_lengths[151])
    # print(word_locations[150] + 1, word_locations[151] + 1)
    return list_with_char_removed, word_locations

# main function for script testing
if(__name__ == '__main__'):
    test_sample = load_article(path, 115035)    
    test_sample = remove_noise(test_sample)
    # print(remove_noise(test_sample)[0])
    # print(remove_noise(test_sample)[1])

    answer_details = {}
    with open('./../new_sample_questions.json', 'r', encoding='utf-8', errors='ignore') as f:
        answer_details = json.load(f)

    answer_char_locs = []
    for q in answer_details['data']:
        answer_begin_pos = q['answer_begin_position '] - 1 # array index => (pos - 1)
        answer_end_pos = q['answer_end_position'] - 1 # array index => (pos - 1)
        answer_char_locs.append((answer_begin_pos, answer_end_pos))

    # # prototype: convert to original character location
    # ## i loop gets index
    # for i in range(test_sample[1].__len__()):
    #     b = answer_char_locs[0][0]
    #     e = answer_char_locs[0][1] ## ending position of word
    #     if(test_sample[1][i] == e):
    #         word_number = i
    #         print('preprocessed word number:', word_number)
    #         print('begin character location:', b, '| ending character location:', e)
    #         print('origin begin character location:', b + 1, '| original ending character location:', e + 1)
    #         ## j loop gets answer
    #         for j in range(test_sample[0].__len__()):
    #             if(word_number == test_sample[0][j][0]):
    #                 print('word:', test_sample[0][j][1])
