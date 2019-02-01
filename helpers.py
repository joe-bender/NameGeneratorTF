"""Helper functions to convert data between various types/representations"""
import string
import numpy as np

# create dictionaries to convert between characters and ints
chars = list(string.ascii_lowercase+' ') # underscore represents end of name
ints = range(27)
char_to_i = {char: i for char, i in zip(chars, ints)}
i_to_char = {i: char for char, i in char_to_i.items()}

def name_to_x(name, sequence_length):
    name_padded = name.lower().ljust(sequence_length)
    name_ints = [char_to_i[char] for char in name_padded]
    return np.array([name_ints])
