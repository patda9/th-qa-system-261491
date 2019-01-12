# coding=utf8

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
from pprint import pprint

def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',fontproperties=fp)
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()

def display_N_words_tsnescatterplot(model,n):
    arr = np.empty((0, 100), dtype='f')
    word_labels = []

    # add the vector for each words to the array
    tmp = 0
    words = model.wv.vocab
    for w in words:
        wrd_vector = model.wv[w]
        word_labels.append(w)
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        tmp+=1
        print(tmp)
        if tmp >= n :
            break
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    tmp = 0
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',fontproperties=fp)
        tmp += 1
        print(tmp)
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()

fp = mpl.font_manager.FontProperties(family='JasmineUPC',size=10)
model = Word2Vec.load("E:\CPE#Y4\databaseTF\w2v_model\w2v_all_corpus.bin")
# display_closestwords_tsnescatterplot(model, 'ชาย')
display_N_words_tsnescatterplot(model,3000)
pprint(model.wv.vocab.__len__())