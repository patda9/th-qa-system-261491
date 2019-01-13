import json
import numpy as np
import preprocessing as prep

# TODO
# load question samples
# load answer samples
# load answer docs
# load answer positions

n = 20
sentences = []

for i in range(len(sentences)):
    sentences[i] = [sentences[i][j * n:(j + 1) * n] for j in range((len(sentences[i]) + n - 1) // n)]
print(sentences)

# from keras.models import Sequential

model = None