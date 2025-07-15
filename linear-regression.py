import numpy as np

class LinearRegression:
    """
    A simple linear regression implementation with support for both analytical and gradient descent solutions.
    
    This class implements linear regression using either:
    - Analytical solution (default): Uses the normal equation (X^T X)^(-1) X^T y
    - Gradient descent: Iteratively updates parameters using gradient descent
    
    Parameters:
        solver (str, optional): The solver to use. 'gd' for gradient descent, None for analytical solution.
        learning_rate (float): Learning rate for gradient descent (default: 0.01)
        n_iterations (int): Number of iterations for gradient descent (default: 1000)
    
    Attributes:
        weights (ndarray): The learned weights/coefficients
        bias (float): The learned bias/intercept term
    """
    
    def __init__(self, solver=None, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.solver = solver

    def fit(self, X, y):
        """
        Fit the linear regression model to the data.
        
        Parameters:
            X (ndarray): Training features of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples,)
        """
        n, p = X.shape
        # Initialize parameters
        self.weights = np.zeros(p)
        self.bias = 0
        
        if self.solver == 'gd':
            # Use gradient descent optimization
            for _ in range(self.n_iterations):
                self.weights, self.bias = self.gradient_descent(X, y, self.weights, self.bias, self.learning_rate)
        else:
            # Use analytical solution (normal equation)
            # Add bias term by augmenting X with a column of ones
            X_augmented = np.hstack((np.ones((n, 1)), X))
            # Solve: beta = (X^T X)^(-1) X^T y
            beta = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
            # Extract weights and bias from solution
            self.weights = beta[1:]
            self.bias = beta[0]

    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
            X (ndarray): Features to predict on
            
        Returns:
            ndarray: Predicted values
        """
        return X @ self.weights + self.bias
    
    def gradient_descent(self, X, y, weights, bias, learning_rate):
        """
        Perform one step of gradient descent to update weights and bias.
        
        Parameters:
            X (ndarray): Training features
            y (ndarray): Target values
            weights (ndarray): Current weights
            bias (float): Current bias
            learning_rate (float): Learning rate for updates
            
        Returns:
            tuple: Updated weights and bias
        """
        n = len(y)
        # Compute predictions
        predictions = X @ weights + bias
        # Compute error (residuals)
        error = predictions - y
        
        # Compute gradients for weights and bias
        weight_gradients = (2/n) * X.T @ error
        bias_gradient = (2/n) * np.sum(error)
        
        # Update parameters using gradient descent
        weights = weights - learning_rate * weight_gradients
        bias = bias - learning_rate * bias_gradient
        
        return weights, bias