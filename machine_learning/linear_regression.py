"""
Linear Regression implementation from scratch using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression using gradient descent optimization.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors
        y : array-like, shape = [n_samples]
            Target values
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_predicted = self.predict(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_predicted)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors
            
        Returns:
        --------
        array-like : predictions
        """
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, y_true, y_pred):
        """
        Compute the mean squared error cost function.
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values
            
        Returns:
        --------
        float : cost value
        """
        n_samples = len(y_true)
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y_true) ** 2)
        return cost
    
    def plot_cost_history(self):
        """Plot the cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
    
    def score(self, X, y):
        """
        Calculate R-squared score.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples
        y : array-like, shape = [n_samples]
            True values
            
        Returns:
        --------
        float : R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


if __name__ == "__main__":
    # Example usage
    from datasets.sample_data import generate_regression_data
    
    # Generate sample data
    X, y = generate_regression_data(n_samples=100, n_features=1, noise=0.1)
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate accuracy
    r2_score = model.score(X, y)
    print(f"R-squared Score: {r2_score:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y, alpha=0.6, label='Actual')
    plt.plot(X[:, 0], predictions, 'r-', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()