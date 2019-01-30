import tensorflow as tf
import helpers
import string

sequence_length = 15
model = tf.keras.models.load_model('models/model.h5')

def name_gen(first_letter):
    name = first_letter
    for i_seq in range(sequence_length):
        x_test = helpers.name_to_x(name, sequence_length)
        y_pred = model.predict(x_test)
        name_ints = y_pred.argmax(axis=2)[0]
        letter = helpers.i_to_char[name_ints[i_seq]]
        if letter == ' ':
            break
        name += letter
    name = name.capitalize().rstrip()
    return name

for letter in string.ascii_lowercase:
    name = name_gen(letter)
    print('{}'.format(name))
