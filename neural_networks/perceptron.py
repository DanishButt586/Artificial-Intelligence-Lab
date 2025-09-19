"""
Perceptron implementation from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    Perceptron classifier implementation.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for weight updates
    n_iterations : int, default=1000
        Number of iterations for training
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors = []
    
    def fit(self, X, y):
        """
        Train the perceptron.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors
        y : array-like, shape = [n_samples]
            Target values (should be -1 or 1)
        """
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to -1 and 1 if they are 0 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Training loop
        for i in range(self.n_iterations):
            errors = 0
            
            for idx, x_i in enumerate(X):
                # Calculate linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Apply activation function (step function)
                y_predicted = self.activation_function(linear_output)
                
                # Update weights and bias if prediction is wrong
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
                # Count errors
                errors += int(update != 0.0)
            
            self.errors.append(errors)
            
            # Stop if no errors
            if errors == 0:
                break
    
    def activation_function(self, x):
        """Step activation function."""
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X):
        """
        Make predictions using the trained perceptron.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors
            
        Returns:
        --------
        array-like : predictions
        """
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.activation_function(linear_output)
        return predictions
    
    def plot_decision_boundary(self, X, y, title="Perceptron Decision Boundary"):
        """
        Plot decision boundary for 2D data.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, 2]
            Input features (must be 2D)
        y : array-like, shape = [n_samples]
            Target labels
        title : str
            Plot title
        """
        if X.shape[1] != 2:
            print("Decision boundary plotting is only supported for 2D data")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        y_ = np.where(y <= 0, -1, 1)
        plt.scatter(X[y_ == 1, 0], X[y_ == 1, 1], c='red', marker='o', label='Class 1', alpha=0.7)
        plt.scatter(X[y_ == -1, 0], X[y_ == -1, 1], c='blue', marker='s', label='Class -1', alpha=0.7)
        
        # Plot decision boundary
        if self.weights is not None:
            # Calculate decision boundary line: w1*x1 + w2*x2 + b = 0
            # x2 = (-w1*x1 - b) / w2
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            
            if abs(self.weights[1]) > 1e-6:  # Avoid division by zero
                x1_line = np.array([x_min, x_max])
                x2_line = (-self.weights[0] * x1_line - self.bias) / self.weights[1]
                plt.plot(x1_line, x2_line, 'k-', linewidth=2, label='Decision Boundary')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_errors(self):
        """Plot the number of errors over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('Number of Errors')
        plt.title('Perceptron Learning Curve')
        plt.grid(True)
        plt.show()
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples
        y : array-like, shape = [n_samples]
            True labels
            
        Returns:
        --------
        float : accuracy score
        """
        predictions = self.predict(X)
        y_ = np.where(y <= 0, -1, 1)
        accuracy = np.mean(predictions == y_)
        return accuracy


if __name__ == "__main__":
    from datasets.sample_data import generate_classification_data
    
    # Generate linearly separable data
    np.random.seed(42)
    X, y = generate_classification_data(n_samples=100, n_features=2, n_classes=2)
    
    # Convert labels to -1 and 1
    y = np.where(y == 0, -1, 1)
    
    # Create and train perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)
    perceptron.fit(X, y)
    
    # Make predictions
    predictions = perceptron.predict(X)
    
    # Calculate accuracy
    accuracy = perceptron.score(X, y)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Converged after {len(perceptron.errors)} iterations")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Decision boundary
    plt.subplot(1, 2, 1)
    perceptron.plot_decision_boundary(X, y, "Perceptron Classification")
    
    # Learning curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Errors')
    plt.title('Perceptron Learning Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()