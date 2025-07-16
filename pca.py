import numpy as np

def pca(X, n_components):
    """
    Perform Principal Component Analysis (PCA) on the input data.
    
    PCA is a dimensionality reduction technique that finds the directions of maximum
    variance in the data and projects the data onto these principal components.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int
        Number of principal components to return (must be <= n_features)
    
    Returns:
    --------
    numpy.ndarray
        Transformed data matrix of shape (n_samples, n_components)
        The data projected onto the top n_components principal components
    
    Notes:
    ------
    The algorithm follows these steps:
    1. Center the data by subtracting the mean
    2. Compute the covariance matrix
    3. Find eigenvalues and eigenvectors of the covariance matrix
    4. Sort by eigenvalues in descending order
    5. Select the top n_components eigenvectors
    6. Project the centered data onto the selected eigenvectors
    """
    
    # Step 1: Center the data by subtracting the mean of each feature
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Step 2: Compute the covariance matrix of the centered data
    # rowvar=False means each column represents a feature
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Perform eigendecomposition to find eigenvalues and eigenvectors
    # eigh() is more efficient for symmetric matrices like covariance matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    # [::-1] reverses the order to get descending instead of ascending
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 5: Select the top n_components eigenvectors (principal components)
    # These represent the directions of maximum variance
    selected_eigenvectors = eigenvectors[:, :n_components]
    
    # Step 6: Project the centered data onto the principal components
    # This transforms the data to the new lower-dimensional space
    return X_centered @ selected_eigenvectors
    
    