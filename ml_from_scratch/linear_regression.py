import numpy as np

class LinearRegression():
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def compute_cost(self, X, y):
        m = X.shape[0]
        err = self.predict(X) - y
        return np.dot(err,err) / (2*m)

    def compute_gradient(self, X, y):
        m = X.shape[0]
        err = self.predict(X) - y
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
        yhat = self.predict(X)
        total_cost = 0
        # Loop over loss for every data in X
        for i in range(m):
            loss_i = (y[i] - yhat[i])**2
            total_cost += loss_i
        total_cost /= 2*m
        return total_cost

    def compute_gradient_loop(self, X, y):
        m, n = X.shape
        yhat = self.predict(X)
        dw = np.zeros(n)
        db = 0.
        for i in range(m):
            err_i = yhat[i] - y[i]
            for j in range(n):
                dw[j] += err_i * X[i][j]
            db += err_i
        return dw / m, db / m