import numpy as np

class KMeans:
    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        
        for _ in range(self.max_iterations):
            self.labels = self._assign(X)
            # Update centroids to the mean of the assigned data points
            self.centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])

    def _assign(self, X):
        # Calculate the distance between each data point and each centroid
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        # Assign each data point to the nearest centroid
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        return self._assign(X)