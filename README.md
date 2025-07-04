# -Clustering-with-K-Means
 Perform unsupervised learning with K-Means clustering.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate and visualize dataset
# Create synthetic dataset with 4 clusters
X, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=1, random_state=42)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize original dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.5)
plt.title('Original Dataset (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('original_dataset.png')
plt.close()

# 2. Fit K-Means and assign cluster labels
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# 3. Elbow Method to find optimal K
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_curve.png')
plt.close()

# 4. Visualize clusters with color-coding
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering Results (K=4)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.savefig('clustering_results.png')
plt.close()

# 5. Evaluate clustering using Silhouette Score
silhouette_avg = silhouette_score(X, cluster_labels)
print(f'Silhouette Score for K=4: {silhouette_avg:.3f}')

# Evaluate Silhouette Score for different K values
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 10), silhouette_scores, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.savefig('silhouette_scores.png')
plt.close()

# Save dataset to CSV
df['Cluster'] = cluster_labels
df.to_csv('clustered_dataset.csv', index=False)
