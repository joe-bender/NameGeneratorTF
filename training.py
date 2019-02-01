import tensorflow as tf
import numpy as np
import string
import helpers
import data

names = data.txt_to_names('txt/yob2017.txt')
x, y, sequence_length = data.names_to_xy(names)

num_chars = 27
embed_size = 128
hidden_length = 128
output_length = num_chars
batch_size = 64
epochs = 1
learning_rate = .01
validation_split = 0

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(num_chars, embed_size, input_length=sequence_length),
    tf.keras.layers.LSTM(hidden_length, return_sequences=True),
    tf.keras.layers.Dense(output_length, activation='softmax'),
])

adam = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

for _ in range(2):
    model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=True,
        validation_split=validation_split)
    model.save('models/model.h5')
