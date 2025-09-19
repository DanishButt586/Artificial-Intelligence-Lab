"""
K-Means Clustering implementation from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    Parameters:
    -----------
    k : int
        Number of clusters
    max_iters : int, default=100
        Maximum number of iterations
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, k, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        
        # Initialize clusters
        self.centroids = None
        self.clusters = None
        
    def fit(self, X):
        """
        Fit K-means clustering to X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        self.centroids = np.random.uniform(
            np.min(X, axis=0), 
            np.max(X, axis=0), 
            size=(self.k, X.shape[1])
        )
        
        for i in range(self.max_iters):
            # Assign points to closest centroid
            self.clusters = self._create_clusters(X)
            
            # Store old centroids
            old_centroids = self.centroids.copy()
            
            # Calculate new centroids
            self.centroids = self._get_centroids(X)
            
            # Check for convergence
            if self._is_converged(old_centroids):
                break
    
    def predict(self, X):
        """
        Predict cluster labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors
            
        Returns:
        --------
        array-like : cluster labels
        """
        distances = self._calculate_distance(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _create_clusters(self, X):
        """Assign each point to the closest centroid."""
        clusters = [[] for _ in range(self.k)]
        
        for point_idx, point in enumerate(X):
            closest_centroid = self._closest_centroid(point)
            clusters[closest_centroid].append(point_idx)
        
        return clusters
    
    def _closest_centroid(self, point):
        """Find the closest centroid to a point."""
        distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
        return np.argmin(distances)
    
    def _get_centroids(self, X):
        """Calculate new centroids as the mean of assigned points."""
        centroids = np.zeros((self.k, X.shape[1]))
        
        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster:
                cluster_mean = np.mean(X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        
        return centroids
    
    def _is_converged(self, old_centroids):
        """Check if centroids have converged."""
        distances = [self._euclidean_distance(old_centroids[i], self.centroids[i]) 
                    for i in range(self.k)]
        return sum(distances) == 0
    
    def _euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _calculate_distance(self, X, centroids):
        """Calculate distance from each point to each centroid."""
        distances = np.zeros((X.shape[0], len(centroids)))
        
        for i, point in enumerate(X):
            for j, centroid in enumerate(centroids):
                distances[i][j] = self._euclidean_distance(point, centroid)
        
        return distances
    
    def plot_clusters(self, X):
        """Plot the clusters and centroids."""
        if X.shape[1] != 2:
            print("Plotting is only supported for 2D data")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'brown']
        
        for i, cluster in enumerate(self.clusters):
            if cluster:
                cluster_points = X[cluster]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i+1}')
        
        # Plot centroids
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                  c='black', marker='x', s=100, linewidths=3, label='Centroids')
        
        ax.set_title(f'K-Means Clustering (k={self.k})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True)
        plt.show()
    
    def inertia(self, X):
        """Calculate within-cluster sum of squares (inertia)."""
        inertia_value = 0
        
        for i, cluster in enumerate(self.clusters):
            if cluster:
                cluster_points = X[cluster]
                centroid = self.centroids[i]
                
                # Sum of squared distances from centroid
                distances = [self._euclidean_distance(point, centroid) ** 2 
                           for point in cluster_points]
                inertia_value += sum(distances)
        
        return inertia_value


if __name__ == "__main__":
    from datasets.sample_data import generate_clustering_data
    
    # Generate sample data
    X, true_labels = generate_clustering_data(n_samples=300, centers=3, random_state=42)
    
    # Apply K-means clustering
    kmeans = KMeans(k=3, max_iters=100, random_state=42)
    kmeans.fit(X)
    
    # Get predictions
    predicted_labels = kmeans.predict(X)
    
    # Calculate inertia
    inertia = kmeans.inertia(X)
    print(f"Inertia: {inertia:.2f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data with true labels
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, alpha=0.6)
    axes[0].set_title('True Clusters')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True)
    plt.colorbar(scatter1, ax=axes[0])
    
    # K-means results
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=predicted_labels, alpha=0.6)
    axes[1].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                   c='black', marker='x', s=100, linewidths=3, label='Centroids')
    axes[1].set_title('K-Means Clustering Results')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    axes[1].grid(True)
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()