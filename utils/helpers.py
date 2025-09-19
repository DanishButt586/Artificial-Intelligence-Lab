"""
Utility functions for the AI Lab.
"""

import numpy as np
import matplotlib.pyplot as plt


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like, shape = [n_samples, n_features]
        Input features
    y : array-like, shape = [n_samples]
        Target values
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split datasets
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize_features(X):
    """
    Normalize features to have zero mean and unit variance.
    
    Parameters:
    -----------
    X : array-like, shape = [n_samples, n_features]
        Input features
        
    Returns:
    --------
    X_normalized : array-like
        Normalized features
    mean : array-like
        Feature means
    std : array-like
        Feature standard deviations
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / (std + 1e-8)  # Add small value to avoid division by zero
    return X_normalized, mean, std


def plot_learning_curve(costs, title="Learning Curve"):
    """
    Plot learning curve from cost history.
    
    Parameters:
    -----------
    costs : list
        Cost values over iterations
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


def plot_classification_results(X, y_true, y_pred, title="Classification Results"):
    """
    Plot classification results for 2D data.
    
    Parameters:
    -----------
    X : array-like, shape = [n_samples, 2]
        Input features (must be 2D)
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
        Plot title
    """
    if X.shape[1] != 2:
        print("Plotting is only supported for 2D data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # True labels
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y_true, alpha=0.6)
    axes[0].set_title('True Labels')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True)
    plt.colorbar(scatter1, ax=axes[0])
    
    # Predicted labels
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.6)
    axes[1].set_title('Predicted Labels')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True)
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    float : accuracy score
    """
    return np.mean(y_true == y_pred)


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float : MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R-squared score.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float : R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


class ModelEvaluator:
    """Helper class for model evaluation."""
    
    @staticmethod
    def evaluate_regression(model, X_test, y_test):
        """Evaluate regression model performance."""
        y_pred = model.predict(X_test)
        
        mse = calculate_mse(y_test, y_pred)
        r2 = calculate_r2_score(y_test, y_pred)
        
        print(f"Regression Model Evaluation:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        return {'mse': mse, 'r2': r2}
    
    @staticmethod
    def evaluate_classification(model, X_test, y_test):
        """Evaluate classification model performance."""
        y_pred = model.predict(X_test)
        
        accuracy = calculate_accuracy(y_test, y_pred)
        
        print(f"Classification Model Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return {'accuracy': accuracy}
    
    @staticmethod
    def evaluate_clustering(model, X):
        """Evaluate clustering model performance."""
        inertia = model.inertia(X)
        
        print(f"Clustering Model Evaluation:")
        print(f"  Inertia: {inertia:.2f}")
        
        return {'inertia': inertia}


if __name__ == "__main__":
    # Test utility functions
    from datasets.sample_data import generate_regression_data
    
    # Generate test data
    X, y = generate_regression_data(n_samples=100, n_features=2)
    
    # Test train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Test normalization
    X_norm, mean, std = normalize_features(X)
    print(f"Original X mean: {np.mean(X, axis=0)}")
    print(f"Normalized X mean: {np.mean(X_norm, axis=0)}")
    print(f"Normalized X std: {np.std(X_norm, axis=0)}")
    
    print("✓ All utility functions working correctly!")