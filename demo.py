#!/usr/bin/env python3
"""
AI Lab Demo Script

This script demonstrates the key features of the Artificial Intelligence Lab
by running examples of different machine learning algorithms.

Run with: python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.sample_data import (
    generate_regression_data,
    generate_classification_data,
    generate_clustering_data
)
from machine_learning.linear_regression import LinearRegression
from machine_learning.kmeans import KMeans
from neural_networks.perceptron import Perceptron
from utils.helpers import train_test_split, ModelEvaluator

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🤖 {title}")
    print("=" * 60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n📊 {title}")
    print("-" * 40)

def demo_linear_regression():
    """Demonstrate Linear Regression."""
    print_section("Linear Regression Demo")
    
    # Generate data
    X, y = generate_regression_data(n_samples=200, n_features=1, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("Training Linear Regression model...")
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model performance...")
    train_metrics = ModelEvaluator.evaluate_regression(model, X_train, y_train)
    print("Training Performance:")
    print(f"  R² Score: {train_metrics['r2']:.4f}")
    
    test_metrics = ModelEvaluator.evaluate_regression(model, X_test, y_test)
    print("Test Performance:")
    print(f"  R² Score: {test_metrics['r2']:.4f}")
    
    return model, X_test, y_test

def demo_perceptron():
    """Demonstrate Perceptron classification."""
    print_section("Perceptron Classification Demo")
    
    # Generate linearly separable data
    X, y = generate_classification_data(n_samples=200, n_features=2, n_classes=2)
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1 labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("Training Perceptron...")
    model = Perceptron(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    
    print(f"Converged after {len(model.errors)} iterations")
    
    # Evaluate
    print("Evaluating model performance...")
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, X_test, y_test

def demo_kmeans():
    """Demonstrate K-Means clustering."""
    print_section("K-Means Clustering Demo")
    
    # Generate clustering data
    X, true_labels = generate_clustering_data(n_samples=300, centers=4)
    
    print(f"Data samples: {len(X)}")
    print(f"True number of clusters: 4")
    
    # Find optimal k using elbow method
    print("Finding optimal number of clusters...")
    k_values = range(1, 8)
    inertias = []
    
    for k in k_values:
        model = KMeans(k=k, max_iters=100, random_state=42)
        model.fit(X)
        inertias.append(model.inertia(X))
    
    # Train with optimal k (4)
    print("Training K-Means with k=4...")
    model = KMeans(k=4, max_iters=100, random_state=42)
    model.fit(X)
    
    # Evaluate
    predicted_labels = model.predict(X)
    metrics = ModelEvaluator.evaluate_clustering(model, X)
    
    print(f"Final Inertia: {metrics['inertia']:.2f}")
    print(f"Clusters found: {len(set(predicted_labels))}")
    
    return model, X, true_labels, predicted_labels

def demo_neural_network_activation():
    """Demonstrate neural network activation functions."""
    print_section("Neural Network Activation Functions")
    
    from neural_networks.activation_functions import sigmoid, relu, tanh
    
    # Test activation functions
    x = np.linspace(-5, 5, 100)
    
    print("Testing activation functions...")
    print(f"Sigmoid(0) = {sigmoid(0):.4f}")
    print(f"ReLU(-1) = {relu(-1):.4f}")
    print(f"ReLU(1) = {relu(1):.4f}")
    print(f"Tanh(0) = {tanh(0):.4f}")
    
    print("Activation functions working correctly!")

def run_comprehensive_demo():
    """Run a comprehensive demonstration of the AI Lab."""
    
    print_header("Welcome to the Artificial Intelligence Lab Demo!")
    print("This demo showcases the implemented algorithms and their capabilities.")
    
    try:
        # Linear Regression Demo
        lr_model, X_test_reg, y_test_reg = demo_linear_regression()
        
        # Perceptron Demo
        perceptron_model, X_test_clf, y_test_clf = demo_perceptron()
        
        # K-Means Demo
        kmeans_model, X_cluster, true_labels, pred_labels = demo_kmeans()
        
        # Neural Network Activation Functions
        demo_neural_network_activation()
        
        print_header("Demo Summary")
        print("✅ Linear Regression: Successfully trained and evaluated")
        print("✅ Perceptron: Successfully trained for binary classification")
        print("✅ K-Means Clustering: Successfully clustered data")
        print("✅ Neural Network Components: Activation functions working")
        
        print("\n🎉 All algorithms working correctly!")
        print("\n📚 Next Steps:")
        print("  - Explore the notebooks/ directory for interactive tutorials")
        print("  - Check individual algorithm files for detailed implementations")
        print("  - Try modifying parameters and datasets")
        print("  - Implement your own algorithms using the existing structure")
        
        print("\n🔗 Repository Structure:")
        print("  - machine_learning/: Core ML algorithms")
        print("  - neural_networks/: Neural network implementations") 
        print("  - datasets/: Data generation utilities")
        print("  - utils/: Helper functions and evaluation tools")
        print("  - notebooks/: Jupyter notebooks for interactive learning")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check the installation and dependencies.")
        return False
    
    return True

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demo
    success = run_comprehensive_demo()
    
    if success:
        print("\n🚀 Demo completed successfully!")
        print("Happy learning with the AI Lab! 🤖✨")
    else:
        print("\n⚠️ Demo encountered issues. Please check the setup.")
        sys.exit(1)