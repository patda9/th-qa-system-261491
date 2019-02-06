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

"""
output: preprocessed (remove noises from each token) arrays of tokens in each article 
        <arrays of tokens in each article: array like>
input: arrays of tokens <arrays of tokens: array like>
"""
# r"[^\u0E00-\u0E7F^0-9^ \t^.]" this pattern removes '.', ',', spaces, tabs and english characters
def remove_noise(array_of_tokens, preprocessing_pattern=re.compile(r"[^\u0E00-\u0E7F^0-9^ ^\t]|^[\u0E00-\u0E7F].[\u0E00-\u0E7F].")):
    thai_number = ['\u0E50', '\u0E51', '\u0E52', '\u0E53', '\u0E54', '\u0E55', '\u0E56', '\u0E57', '\u0E58', '\u0E59']
    
    original_token_lengths = []
    for word in array_of_tokens:
        original_token_lengths.append(word.__len__()) # for each j => temp contains each article's word lengths

    # this below block removes any charater in regex_pattern
    original_token_indexes = []
    tokens_with_chars_removed = []
    for i in range(array_of_tokens.__len__()):
        chars_to_remove = re.findall(preprocessing_pattern, array_of_tokens[i])
        temp = '' # declare for characters that pass condition
        for j in range(array_of_tokens[i].__len__()):
            if(not(array_of_tokens[i][j] in chars_to_remove)):
                if(array_of_tokens[i][j] in thai_number):
                    if(array_of_tokens[i][j] == thai_number[0]):
                        temp += '0'
                    if(array_of_tokens[i][j] == thai_number[1]):
                        temp += '1'
                    if(array_of_tokens[i][j] == thai_number[2]):
                        temp += '2'
                    if(array_of_tokens[i][j] == thai_number[3]):
                        temp += '3'
                    if(array_of_tokens[i][j] == thai_number[4]):
                        temp += '4'
                    if(array_of_tokens[i][j] == thai_number[5]):
                        temp += '5'
                    if(array_of_tokens[i][j] == thai_number[6]):
                        temp += '6'
                    if(array_of_tokens[i][j] == thai_number[7]):
                        temp += '7'
                    if(array_of_tokens[i][j] == thai_number[8]):
                        temp += '8'
                    if(array_of_tokens[i][j] == thai_number[9]):
                        temp += '9'
                else:
                    temp += array_of_tokens[i][j] # concatenate charaters those are not in chars_to_remove
        # this below condition filters remaining single \n, \t, spaces commas and dots tokens
        if(temp.__len__() and not(temp in re.findall(r"^\s|^,|^.", temp))):
            original_token_indexes.append(i)
            tokens_with_chars_removed.append(temp) # append temp(word <string>) to tokens_with_chars_remove
    
    # print(tokens_with_chars_removed) # TESTING FUNCTION

    summation = 0
    for i in range(original_token_lengths.__len__()):
        summation += original_token_lengths[i]
        original_token_lengths[i] = summation
    
    # print(original_token_lengths)
    # print(original_token_indexes)

    # this below block gives the remaining word's original position ranges
    # in format range(0, 10) => 0..9
    token_ranges = []
    for i in range(1, original_token_indexes.__len__() - 1):
        start = original_token_indexes[i-1]
        end = start + 1
        # token_ranges.append((original_token_lengths[start-1], original_token_lengths[end]))
        if(start):
            # append range of each token
            token_ranges.append((original_token_lengths[start - 1], original_token_lengths[end]))
        else:
            token_ranges.append((original_token_lengths[start], original_token_lengths[end]))
    
    # print(token_ranges)
    # print(selected_plain_text_article[0][528:537]) # answer position form => (start+1, end)
    
    """
    for i in range(token_ranges.__len__()): # TESTING FUNCTION: returns tokens string in 1 line
        temp += selected_plain_text_article[0][token_ranges[i][0]:token_ranges[i][1] - 1] + ' '
    print(temp)
    """

    # preprocessing gives remaining tokens, their old indexes after tokenized
    # and their position ranges(range canbe converted to length)
    return [tokens_with_chars_removed, original_token_indexes, original_token_lengths]

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
