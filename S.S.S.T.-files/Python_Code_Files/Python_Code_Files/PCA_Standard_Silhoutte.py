#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:10:03 2024

@author: t.munkhsaruul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # Import StandardScaler instead of RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Sample data (replace this with your actual dataset)
# data = pd.read_csv('your_dataset.csv')

data = pd.read_csv('cleaned_data_clustering.csv')

# Step 1: Add new columns (average of Previous and Repayment)
data['Avg_Previous'] = data[['Previous 1', 'Previous 2', 'Previous 3', 'Previous 4', 'Previous 5', 'Previous 6']].mean(axis=1)
data['Avg_Repayment'] = data[['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']].mean(axis=1)
data['Avg_Amount'] = data[['Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6']].mean(axis=1)
repayment_columns = ['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']
data['Mode Repayment Status'] = data[repayment_columns].mode(axis=1)[0]
# Keep a copy of the original data for descriptive statistics
original_data = data.copy()

# Step 2: Define columns to scale
columns_to_scale = ['Credit', 'Age', 'Previous 1', 'Previous 2', 'Previous 3', 
                    'Previous 4', 'Previous 5', 'Previous 6', 'Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6']

# Function to scale, apply PCA, and cluster
def scale_and_cluster(scaler, data, columns_to_scale, n_clusters):
    # Scale numerical features
    scaled_features = scaler.fit_transform(data[columns_to_scale])
    scaled_data = data.copy()
    scaled_data[columns_to_scale] = scaled_features

    # Dimensionality Reduction (PCA)
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    pca_features = pca.fit_transform(scaled_features)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pca_features)

    # Assign the cluster labels back to the original data
    data['Cluster'] = kmeans.labels_

    # Return the cluster labels and PCA features for silhouette score calculation and plotting
    return data, pca_features, kmeans.labels_

# Step 3: Determine the optimal number of clusters using silhouette analysis
def find_optimal_clusters_silhouette(scaler, data, columns_to_scale):
    silhouette_scores = []
    cluster_range = range(2, 11)  # Adjust the range based on your preference

    for n_clusters in cluster_range:
        _, pca_features, cluster_labels = scale_and_cluster(scaler, data, columns_to_scale, n_clusters)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(pca_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot Silhouette Scores to choose the optimal number of clusters
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Number of Clusters (StandardScaler)')
    plt.show()
    
    # Return the cluster range and silhouette scores for further analysis
    return cluster_range, silhouette_scores

# Call the silhouette score function for StandardScaler
standard_cluster_range, standard_silhouette_scores = find_optimal_clusters_silhouette(StandardScaler(), data, columns_to_scale)

# Determine the optimal number of clusters based on silhouette score
optimal_clusters = standard_cluster_range[np.argmax(standard_silhouette_scores)]

# Perform clustering with the optimal number of clusters using StandardScaler
original_data, pca_features, cluster_labels = scale_and_cluster(StandardScaler(), data, columns_to_scale, optimal_clusters)

# Evaluate clustering
def evaluate_clustering(pca_features, cluster_labels):
    db_index = round(davies_bouldin_score(pca_features, cluster_labels), 3)
    s_score = round(silhouette_score(pca_features, cluster_labels), 3)
    ch_index = round(calinski_harabasz_score(pca_features, cluster_labels), 3)
    print(f"Davies-Bouldin Index: {db_index}")
    print(f"Silhouette Score: {s_score}")
    print(f"Calinski Harabasz Index: {ch_index}")
    return db_index, s_score, ch_index

db_kmeans, ss_kmeans, ch_kmeans = evaluate_clustering(pca_features, cluster_labels)

# Cluster profiling
cluster_profile = original_data.groupby('Cluster').mean()
print(cluster_profile)

# Scatter plot of the PCA-reduced features
plt.figure(figsize=(12, 8))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, cmap='viridis', marker='o')
plt.title('KMeans Clustering Results')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# Count the number of points in each cluster
cluster_counts = original_data['Cluster'].value_counts().sort_index()
print(f'Optimal number of clusters: {optimal_clusters}')
print(f'Silhouette Score: {ss_kmeans}')
print('Number of points in each cluster:')
print(cluster_counts)

# Descriptive statistics for selected columns in each cluster without scaling
selected_columns = ['Credit', 'Age', 'Status', 'Education', 'Avg_Previous', 'Avg_Repayment', 'Avg_Amount']
cluster_descriptive_stats = original_data.groupby('Cluster')[selected_columns].describe().transpose()

# Round the descriptive statistics to 2 decimal places
cluster_descriptive_stats_rounded = cluster_descriptive_stats.round(2)
print('\nDescriptive statistics for each cluster (rounded to 2 decimal places):')
print(cluster_descriptive_stats_rounded)

# Save to CSV for detailed examination
cluster_descriptive_stats_rounded.to_csv('cluster_descriptive_stats_selected_columns_rounded.csv')

# Display with better formatting using tabulate
try:
    from tabulate import tabulate
    print(tabulate(cluster_descriptive_stats_rounded, headers='keys', tablefmt='psql'))
except ImportError:
    print(cluster_descriptive_stats_rounded)

# Pairplot visualization for specific columns
selected_columns_for_pairplot = ['Credit', 'Age', 'Status', 'Education', 'Avg_Previous', 'Avg_Repayment', 'Avg_Amount']

sns.pairplot(data=original_data[selected_columns_for_pairplot + ['Cluster']], hue="Cluster", palette='Set1')
plt.show()

# Bar plot for means
means = original_data.groupby('Cluster')[selected_columns].mean().round(2)
means.plot(kind='bar', figsize=(14, 7))
plt.title('Means of Selected Features by Cluster')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.show()

# Box plot for distribution
plt.figure(figsize=(20, 15))
for i, column in enumerate(selected_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Cluster', y=column, data=original_data)
    plt.title(f'Boxplot of {column} by Cluster')
plt.tight_layout()
plt.show()
