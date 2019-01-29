import numpy as np

def txt_to_names(filename):
    with open(filename, 'r') as f:
        return [line.split(',')[0] for line in f.read().split()]

# def names_to_npy(names):
#     npy =
#     np.save('npy/names', npy)
