# utils.py
import string
import numpy as np

ALPHABET = string.ascii_lowercase
A2I = {c:i for i,c in enumerate(ALPHABET)}
I2A = {i:c for c,i in A2I.items()}

def word_to_indices(word, max_len):
    """Return an array shape (max_len,) where each position is 0..25 or -1 for pad."""
    arr = np.full(max_len, -1, dtype=np.int64)
    for i, ch in enumerate(word[:max_len]):
        if ch in A2I:
            arr[i] = A2I[ch]
        else:
            arr[i] = -1
    return arr

def mask_word_from_indices(indices):
    # indices: array of ints or -1
    return ''.join((I2A[i] if i >=0 else '_') for i in indices)

def one_hot_indices(indices, num_classes=26):
    # shape (max_len, num_classes)
    max_len = len(indices)
    mat = np.zeros((max_len, num_classes), dtype=np.float32)
    for i, idx in enumerate(indices):
        if idx >= 0:
            mat[i, idx] = 1.0
    return mat
