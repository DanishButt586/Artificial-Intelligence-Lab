"""
Activation functions for neural networks.
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    
    Parameters:
    -----------
    x : array-like
        Input values
        
    Returns:
    --------
    array-like : activated values
    """
    # Clip x to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Parameters:
    -----------
    x : array-like
        Input values
        
    Returns:
    --------
    array-like : activated values
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU function."""
    return (x > 0).astype(float)


def tanh(x):
    """
    Hyperbolic tangent activation function.
    
    Parameters:
    -----------
    x : array-like
        Input values
        
    Returns:
    --------
    array-like : activated values
    """
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh function."""
    return 1 - np.tanh(x) ** 2


def softmax(x):
    """
    Softmax activation function.
    
    Parameters:
    -----------
    x : array-like
        Input values
        
    Returns:
    --------
    array-like : activated values (probabilities)
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def linear(x):
    """Linear activation function (identity)."""
    return x


def linear_derivative(x):
    """Derivative of linear function."""
    return np.ones_like(x)


# Dictionary mapping activation names to functions and derivatives
ACTIVATION_FUNCTIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative)
}


def get_activation_function(name):
    """
    Get activation function and its derivative by name.
    
    Parameters:
    -----------
    name : str
        Name of the activation function
        
    Returns:
    --------
    tuple : (activation_function, derivative_function)
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test activation functions
    x = np.linspace(-5, 5, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sigmoid
    y_sigmoid = sigmoid(x)
    axes[0, 0].plot(x, y_sigmoid, 'b-', label='sigmoid')
    axes[0, 0].plot(x, sigmoid_derivative(x), 'r--', label='derivative')
    axes[0, 0].set_title('Sigmoid Activation')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # ReLU
    y_relu = relu(x)
    axes[0, 1].plot(x, y_relu, 'b-', label='relu')
    axes[0, 1].plot(x, relu_derivative(x), 'r--', label='derivative')
    axes[0, 1].set_title('ReLU Activation')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Tanh
    y_tanh = tanh(x)
    axes[1, 0].plot(x, y_tanh, 'b-', label='tanh')
    axes[1, 0].plot(x, tanh_derivative(x), 'r--', label='derivative')
    axes[1, 0].set_title('Tanh Activation')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Softmax (for a small subset)
    x_small = np.linspace(-2, 2, 50)
    y_softmax = softmax(x_small.reshape(-1, 1)).flatten()
    axes[1, 1].plot(x_small, y_softmax, 'b-', label='softmax')
    axes[1, 1].set_title('Softmax Activation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()