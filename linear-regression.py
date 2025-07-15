import numpy as np

class LinearRegression:
    def __init__(self, solver=None, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.solver = solver

    def fit(self, X, y):
        n, p = X.shape
        self.weights = np.zeros(p)
        self.bias = 0
        if self.solver == 'gd':
            for _ in range(self.n_iterations):
                self.weights, self.bias = self.gradient_descent(X, y, self.weights, self.bias, self.learning_rate)
        else:
            X_augmented = np.hstack((np.ones((n, 1)), X))
            beta = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
            self.weights = beta[1:]
            self.bias = beta[0]

    def predict(self, X):
        return X @ self.weights + self.bias
    
    def gradient_descent(self, X, y, weights, bias, learning_rate):
        n = len(y)
        predictions = X @ weights + bias
        error = predictions - y
        
        weight_gradients = (2/n) * X.T @ error
        bias_gradient = (2/n) * np.sum(error)
        
        weights = weights - learning_rate * weight_gradients
        bias = bias - learning_rate * bias_gradient
        
        return weights, bias