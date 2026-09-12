import numpy as np
from ml_from_scratch.activations import sigmoid

class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def predict_proba(self, X):
        Z = np.dot(X, self.w) + self.b
        return sigmoid(Z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_cost(self, X, y):
        m = X.shape[0]
        f = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        return -(np.dot(y,np.log(f)) + np.dot((1 - y),np.log(1 - f))) / m

    def compute_gradient(self, X, y):
        m = X.shape[0]
        err = self.predict_proba(X) - y
        dw = np.dot(X.T, err) /m
        db = err.mean()
        return dw, db 

    def fit(self, X, y):
        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history =[]
        for _ in range(self.n_iters):
            dw,db = self.compute_gradient(X,y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            self.cost_history.append(self.compute_cost(X, y))
        return self

    def compute_cost_loop(self, X, y):
        m = X.shape[0]
        f_wb = self.predict_proba(X)
        total_cost = 0
        for i in range(m):
            loss = -(y[i] * np.log(f_wb[i]) + (1-y[i])*np.log(1-f_wb[i]))
            total_cost += loss
        return total_cost /m

    def compute_gradient_loop(self, X, y):
        m, n = X.shape
        yhat = self.predict_proba(X)
        dw = np.zeros(n)
        db = 0.
        for i in range(m):
            err_i = yhat[i] - y[i]
            for j in range(n):
                dw[j] += err_i * X[i][j]
            db += err_i
        return dw / m, db / m