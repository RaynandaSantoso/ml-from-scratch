import numpy as np

class NeuralNetwork():

    # def __init__(self, ):

    def dense(self, a_in, W, b, g):
        z = np.matmul(a_in, W) + b
        a_out = g(z)
        return a_out

    def dense_loop(self, a_in, W, b, g):
        units = W.shape[1]
        a_out = np.zeros(units)
        for j in range(units):
            w = W[:,j]
            z = np.dot(w, a_in) + b[j]
            a_out[j] = g(z)
        return a_out

    def sequential(self, x, layers):
        a = x
        for W, b, g in layers:
            a = self.dense(a, W, b, g)
        f_x = a
        return f_x