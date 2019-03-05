import json
import numpy as np
import os

def vectorize_corpus(path, word_vectors, embedding_shape=(100, )):
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
