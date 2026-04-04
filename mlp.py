import numpy as np
from scipy.special import softmax

class MLP:
    """
    A multi-layer perceptron implementation for classification using gradient descent optimization.
    
    This class implements a multi-layer perceptron model for classification.
    It uses the ReLU activation function for the hidden layer and the softmax function for the output layer.
    """
    
    def __init__(self, input_size, hidden_size, output_size, n_iterations=1000, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.random.randn(output_size)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def _forward(self, X):
        self.layer1 = np.dot(X, self.weights1) + self.bias1
        self.activation = self._relu(self.layer1)
        self.layer2 = np.dot(self.activation, self.weights2) + self.bias2
        self.output = softmax(self.layer2, axis=1)
        return self.output

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _backward(self, X, y, output):
        # Derivative of the cross entropy loss function with respect to the final layer prior to the softmax function
        d_layer2 = (output - y) / X.shape[0]

        d_weights2 = np.dot(self.activation.T, d_layer2)
        d_bias2 = np.sum(d_layer2, axis=0)

        d_activation = np.dot(d_layer2, self.weights2.T)
        d_layer1 = d_activation * self._relu_derivative(self.layer1)

        d_weights1 = np.dot(X.T, d_layer1)
        d_bias1 = np.sum(d_layer1, axis=0)

        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            output = self._forward(X)
            self._backward(X, y, output)

    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)

    def predict_proba(self, X):
        return self._forward(X)