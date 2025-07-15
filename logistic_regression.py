import numpy as np

class BinaryLogisticRegression:
    """
    A binary logistic regression implementation using gradient descent optimization.
    
    This class implements logistic regression for binary classification problems.
    It uses the sigmoid function to model the probability of the positive class
    and includes optional L2 regularization to prevent overfitting.
    
    Parameters:
        learning_rate (float): Learning rate for gradient descent (default: 0.01)
        n_iterations (int): Number of iterations for gradient descent (default: 1000)
        lambda_ (float): L2 regularization parameter (default: 0, no regularization)
    
    Attributes:
        weights (ndarray): The learned weights/coefficients
        bias (float): The learned bias/intercept term
        sigmoid (function): Sigmoid activation function
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_=0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        # Sigmoid function: maps any real number to (0, 1) for probability
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Fit the logistic regression model to the data using gradient descent.
        
        Parameters:
            X (ndarray): Training features of shape (n_samples, n_features)
            y (ndarray): Binary target values of shape (n_samples,) with values 0 or 1
        """
        n, p = X.shape
        # Initialize parameters to zero
        self.weights = np.zeros(p)
        self.bias = 0
        
        # Perform gradient descent for specified number of iterations
        for _ in range(self.n_iterations):
            self.weights, self.bias = self.gradient_descent(X, y, self.weights, self.bias, self.learning_rate, self.lambda_)

    def predict_proba(self, X):
        """
        Predict class probabilities for the samples in X.
        
        Parameters:
            X (ndarray): Features to predict on
            
        Returns:
            ndarray: Probability of positive class for each sample
        """
        # Compute log-odds (linear combination of features)
        log_odds = X @ self.weights + self.bias
        # Apply sigmoid to get probabilities
        return self.sigmoid(log_odds)
    
    def predict(self, X):
        """
        Predict class labels for the samples in X.
        
        Parameters:
            X (ndarray): Features to predict on
            
        Returns:
            ndarray: Predicted class labels (0 or 1)
        """
        # Use 0.5 as threshold for binary classification
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def gradient_descent(self, X, y, weights, bias, learning_rate, lambda_):
        """
        Perform one step of gradient descent to update weights and bias.
        
        Parameters:
            X (ndarray): Training features
            y (ndarray): Binary target values
            weights (ndarray): Current weights
            bias (float): Current bias
            learning_rate (float): Learning rate for updates
            lambda_ (float): L2 regularization parameter
            
        Returns:
            tuple: Updated weights and bias
        """
        n = len(y)
        # Forward pass: compute predicted probabilities
        y_hat = self.predict_proba(X)
        # Compute error (difference between predicted and actual probabilities)
        error = y_hat - y
        
        # Compute gradients with L2 regularization for weights
        weights_gradient = (1/n) * X.T @ error + lambda_ * weights
        bias_gradient = (1/n) * np.sum(error)
        
        # Update parameters using gradient descent
        weights = weights - learning_rate * weights_gradient
        bias = bias - learning_rate * bias_gradient
        
        return weights, bias