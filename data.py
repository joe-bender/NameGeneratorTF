import numpy as np
import helpers

def txt_to_names(filename):
    with open(filename, 'r') as f:
        return [line.split(',')[0] for line in f.read().split()]

def names_to_xy(names):
    sequence_length = len(max(names, key=len))
    x = []
    y = []
    for name in names:
        name_padded = name.lower().ljust(sequence_length + 1)
        name_ints = [helpers.char_to_i[char] for char in name_padded]
        x.append(np.array(name_ints[:-1]))
        y.append(np.array([[char] for char in name_ints[1:]]))
    x = np.stack(x)
    y = np.stack(y)
    return x, y, sequence_length
