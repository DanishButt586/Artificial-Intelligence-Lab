"""
Sample data generation utilities for testing ML algorithms.
"""

import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs


def generate_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=42):
    """
    Generate sample regression data.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples to generate
    n_features : int, default=1
        Number of features
    noise : float, default=0.1
        Noise level in the data
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : array-like, shape = [n_samples, n_features]
        Input features
    y : array-like, shape = [n_samples]
        Target values
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y


def generate_classification_data(n_samples=100, n_features=2, n_classes=2, random_state=42):
    """
    Generate sample classification data.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples to generate
    n_features : int, default=2
        Number of features
    n_classes : int, default=2
        Number of classes
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : array-like, shape = [n_samples, n_features]
        Input features
    y : array-like, shape = [n_samples]
        Target labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state
    )
    return X, y


def generate_clustering_data(n_samples=300, centers=3, n_features=2, random_state=42):
    """
    Generate sample clustering data.
    
    Parameters:
    -----------
    n_samples : int, default=300
        Number of samples to generate
    centers : int, default=3
        Number of cluster centers
    n_features : int, default=2
        Number of features
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : array-like, shape = [n_samples, n_features]
        Input features
    y : array-like, shape = [n_samples]
        Cluster labels
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state
    )
    return X, y


def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """
    Generate non-linear sample data for testing.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples to generate
    noise : float, default=0.1
        Noise level in the data
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : array-like, shape = [n_samples, 1]
        Input features
    y : array-like, shape = [n_samples]
        Target values
    """
    np.random.seed(random_state)
    X = np.random.uniform(-2, 2, (n_samples, 1))
    y = X[:, 0] ** 2 + noise * np.random.randn(n_samples)
    return X, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test all data generation functions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Regression data
    X_reg, y_reg = generate_regression_data(n_samples=100, n_features=1)
    axes[0, 0].scatter(X_reg[:, 0], y_reg, alpha=0.6)
    axes[0, 0].set_title('Regression Data')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].grid(True)
    
    # Classification data
    X_clf, y_clf = generate_classification_data(n_samples=200, n_features=2)
    scatter = axes[0, 1].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, alpha=0.6)
    axes[0, 1].set_title('Classification Data')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].grid(True)
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Clustering data
    X_cluster, y_cluster = generate_clustering_data(n_samples=300, centers=3)
    scatter = axes[1, 0].scatter(X_cluster[:, 0], X_cluster[:, 1], c=y_cluster, alpha=0.6)
    axes[1, 0].set_title('Clustering Data')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    axes[1, 0].grid(True)
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Non-linear data
    X_nonlin, y_nonlin = generate_nonlinear_data(n_samples=100)
    axes[1, 1].scatter(X_nonlin[:, 0], y_nonlin, alpha=0.6)
    axes[1, 1].set_title('Non-linear Data')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()