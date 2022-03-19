import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler

# get the data
data = pd.read_csv("manifold_data.csv",names=["word1","word2","connected","similarity"])

# get the list of input and output data
word1, word2 = [], []
words = []
y = []
for i in range(len(data)):
    word1.append(data["word1"][i])
    word2.append(data["word2"][i])
    words.append(data["word1"][i])
    words.append(data["word2"][i])
    y.append(data["connected"][i])

# create a vocabulary list
long_word = word1[0] + word2[0]
for i in range(1,len(y)):
    long_word = long_word + word1[i] + word2[i]
    
vocab = list(set(long_word))
    
# pad each word with spaces to make them equal length
l = len(max(words, key=len)) 
word1 = [list(s.ljust(l)) for s in word1]
word2 = [list(s.ljust(l)) for s in word2]
        
# create a text layer
layer = tf.keras.layers.StringLookup(vocabulary=vocab)

# map words to integers
word1 = layer(word1)
word2 = layer(word2)

# format words lists
word1 = [np.array(word1[i]) for i in range(len(word1))]
word2 = [np.array(word2[i]) for i in range(len(word2))]

# combine word lists
words = word1+word2

# scale input data
scaler = StandardScaler()
scaler.fit(words)
word1_scaled = scaler.transform(word1)
word2_scaled = scaler.transform(word2)

# split the data into train, validation, and test sets
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(word1_scaled, word2_scaled, y, test_size = 0.2)
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1_train, X2_train, y_train, test_size = 0.1)

# build a model with two input layers
mlp1 = Sequential()
mlp1.add(Dense(100, input_dim=np.array(X1_train).shape[1], activation="relu"))
mlp1.add(Dense(50, activation="relu"))

mlp2 = Sequential()
mlp2.add(Dense(100, input_dim=np.array(X2_train).shape[1], activation="relu"))
mlp2.add(Dense(50, activation="relu"))

# create the input to our final set of layers as the *output* of both MLPs
combinedInput = concatenate([mlp1.output, mlp2.output])

# add fully connected layers
x = Dense(25, activation="relu")(combinedInput)
x = Dense(1, activation="sigmoid")(x)

# define the complete model
model = Model(inputs=[mlp1.input, mlp2.input], outputs=x)

# define the optimiser with learning rate and decay rate
opt = Adam(lr=1e-4)

# compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# fit the model
history = model.fit([np.array(X1_train), np.array(X2_train)], np.array(y_train), batch_size=128, epochs=50, 
          verbose=1, validation_data = ([np.array(X1_val), np.array(X2_val)], np.array(y_val)))

# plot accuracy learning curve
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
# make predictions on the test set 
predictions = np.round(model.predict([np.array(X1_test),np.array(X2_test)]))

# get accuracy score of predictions
accuracy = accuracy_score(y_test,predictions)

print('Accuracy:',accuracy)

