import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    return np.exp(shifted)/np.exp(shifted).sum(axis=-1, keepdims=True)