import tensorflow as tf
import numpy as np
import string
import helpers

sequence_length = 14
num_chars = 27
embed_size = 128
hidden_length = 64
output_length = num_chars
batch_size = 1

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(num_chars, embed_size, input_length=sequence_length),
    tf.keras.layers.LSTM(hidden_length, return_sequences=True),
    tf.keras.layers.Dense(output_length, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

name_in = 'alexander______'
name = [helpers.char_to_i[char] for char in name_in]
x_train = np.array([name[:-1]])
y_train = np.array([[[char] for char in name[1:]]])

# x_train = np.array([[0, 1, 2, 3]])
# y_train = np.array([[[1], [2], [3], [4]]])

model.fit(x_train, y_train, epochs=200)

y_pred = model.predict(x_train)
ints = y_pred.argmax(axis=2)[0]
name_out = ''.join([helpers.i_to_char[i] for i in ints])
print(name_in)
print(name_out)
