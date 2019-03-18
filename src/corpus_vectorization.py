import json
import numpy as np
import os

def vectorize_document(path, output_path, word_vectors, embedding_shape=(100, )):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        document = json.load(f)
    
    for i in range(len(document)):
        try:
            document[i] = word_vectors[document[i]]
        except KeyError:
            document[i] = np.zeros(embedding_shape)
    
    document = np.asarray(document)
    np.save(output_path, document)

    return document

if __name__ == "__main__":
    from gensim.models import Word2Vec
    WV_PATH = 'D:/Users/Patdanai/th-qasys-db/word_vectors_model/word2vec.model'
    word_vectors = Word2Vec.load(WV_PATH)
    embedding_shape = word_vectors['มกราคม'].shape
    print(embedding_shape)

    PATH = 'D:/Users/Patdanai/th-qasys-db/tokenized_wiki_corpus/'
    OUT_PATH = 'D:/Users/Patdanai/th-qasys-db/corpus_wv/'
    
    files = os.listdir(PATH)
    start = 0
    for i in range(start, len(files)):
        f_name = [c for c in files[i] if c.isdigit()]
        f_name = ''.join(f_name)
        vectorize_document(PATH + files[i], OUT_PATH + f_name + '.npy', word_vectors, embedding_shape=embedding_shape)
        print('[%d/%d]\tSaved\t%s\tto\t%s. \r' % (i, len(files), f_name + '.npy', OUT_PATH))
        
    # print(np.load(OUT_PATH + '1.npy').shape)
    
    # with open(PATH + '1.json', encoding='utf-8') as f:
    #     data = json.load(f)
    # data = np.array(data)
    # print(data.shape)
