from gensim.models import Word2Vec

word_vectors = Word2Vec.load('D:/Users/Patdanai/th-qasys-db/word_vectors_model/word2vec.model')
print(word_vectors['มกราคม'].shape)