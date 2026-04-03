import numpy as np

class WeakClf:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.alpha = None
        self.polarity = 1

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators = []
        self.weights = []

    def fit(self, X, y):
        """
        Fit the AdaBoost model to the data.
        
        Parameters:
            X (ndarray): Training features of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples,) where 1 is positive and -1 is negative
        """
        n_samples, n_features = X.shape
        # Initialize weights
        self.weights = np.ones(n_samples) / n_samples

        # Traverse features
        for _ in range(self.n_estimators):

            clf = WeakClf()

            # Store minimum error across all features
            min_error = float('inf')
            
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Try all thresholds for this feature
                for value in unique_values:

                    polarity = 1
                    
                    # Create a binary classifier for this feature and threshold
                    binary_classifier = np.where(feature_values < value, 1, -1)

                    # Calculate error
                    error = np.sum(self.weights[binary_classifier != y])

                    # If error over 50%, flip classification
                    if error > 0.5:
                        binary_classifier = -binary_classifier
                        error = 1 - error
                        polarity = -1

                    # Update minimum error if this error is smaller
                    if error < min_error:
                        min_error = error
                        clf.feature = feature_i
                        clf.threshold = value
                        clf.polarity = polarity

            if (min_error == 0):
                raise ValueError("Min error is 0, cannot train weak classifier")

            # Alpha = 0.5 * log((1 - error) / error)
            clf.alpha = 0.5 * np.log((1 - min_error) / min_error)

            # Update weights
            predictions = np.zeros(n_samples)
            predictions = clf.polarity * np.where(X[:, clf.feature] < clf.threshold, 1, -1)
            self.weights = self.weights * np.exp(- clf.alpha * y * predictions)
            self.weights = self.weights / np.sum(self.weights)

            # Store classifier
            self.estimators.append(clf)

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        for est in self.estimators:
            predictions += est.alpha * est.polarity * np.where(X[:, est.feature] < est.threshold, 1, -1)
        return np.sign(predictions)