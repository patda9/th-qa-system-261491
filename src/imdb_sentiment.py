# get data
from keras.datasets import imdb

vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))
print()

# word database
word2id = imdb.get_word_index() # value to key
id2word = {i: word for word, i in word2id.items()} # key to value
print('---review with words-id---')
print(X_train[0]) # example of x_train
print('---review with words---')
print([id2word.get(i, ' ') for i in X_train[0]]) # convert ids to words in x_train
print('---label---')
print(y_train[0]) # example of y_train
print()

print('Minimum review length: {}'.format(len(min((X_test + X_test), key=len))))
print('Maximum review length: {}'.format(len(max((X_train + X_test), key=len))))
print()

# pad sequence
from keras.preprocessing import sequence

max_words = 500 # to have same shape in each sample by padding below
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print('---example review with padded words-id---')
print(X_train.shape)
print()

# building model
from keras.models import Model
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
embedding_size=64
lstm_output = 192
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(lstm_output))
model.add(Dense(1, activation='sigmoid'))
# drop out regularization
json_model = model.to_json()
print(json_model)
print(model.summary())
exit()

# training and evaluation
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(index=0).output)

intermediate_output = intermediate_layer_model.predict(X_test)
print()
print(intermediate_output)

batch_size = 64
epochs = 1
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
print(X_train2)
print(y_train2.shape)
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs)
model.save('imbd-test-model.h5')
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
print()

# save model 
