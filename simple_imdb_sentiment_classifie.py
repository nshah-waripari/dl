from keras.datasets import imdb
from keras import preprocessing

max_features = 10000 #number of words to consider as a feature
maxlen = 20 #cut off reviews after only 20 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
print(y_train.shape)
# turn the list of integers into 2-d integers tensor of shape (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen = maxlen)

#model layer

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(max_features, 8, input_length = maxlen)) #8 is dimension of the embedding vector

#Flattens the 3d tensor output of Embedding layer into 2d tensor of shape(samples, 8*maxlen)
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

#fit the model
history = model.fit(x_train, y_train, epochs = 10,
                    batch_size = 32,
                    validation_split = 0.2)

#plot accuracy and loss graphs
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) +1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.legend()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.show()
if __name__ == "__main__":
    main()