#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:30:25 2024

@author: t.munkhsaruul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load your dataset
data = pd.read_csv('cleaned_data_clustering.csv')

# Step 1: Add new columns (average of Previous and Repayment)
data['Avg_Previous'] = data[['Previous 1', 'Previous 2', 'Previous 3', 'Previous 4', 'Previous 5', 'Previous 6']].mean(axis=1)
data['Avg_Repayment'] = data[['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']].mean(axis=1)
data['Avg_Amount'] = data[['Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6']].mean(axis=1)

# Handle mode calculation
repayment_columns = ['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']
data['Mode Repayment Status'] = data[repayment_columns].mode(axis=1).iloc[:, 0]

# Keep a copy of the original data for descriptive statistics
original_data = data.copy()

# Step 2: Define columns to scale
columns_to_scale = ['Credit', 'Age', 'Previous 1', 'Previous 2', 'Previous 3', 
                    'Previous 4', 'Previous 5', 'Previous 6', 'Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6']

# Scale numerical features
scaler = RobustScaler()
scaled_features = scaler.fit_transform(data[columns_to_scale])
scaled_data = data.copy()
scaled_data[columns_to_scale] = scaled_features

# Dimensionality Reduction (PCA)
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_features = pca.fit_transform(scaled_features)

# DBSCAN Clustering with multiple values for eps and min_samples
eps_values = [0.6, 0.7, 0.8]
min_samples_values = [3, 4, 5]

best_score = -1
best_params = {}
best_dbscan = None
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(pca_features)
        labels = dbscan.labels_
        
        # Check if there are at least 2 clusters (excluding noise)
        if len(set(labels)) > 1:
            score = silhouette_score(pca_features, labels)
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_dbscan = dbscan
                best_labels = labels

print(f"Best DBSCAN params: {best_params}, Silhouette Score: {best_score}")

# Assign the best cluster labels back to the original data
data['Cluster'] = best_labels

# Evaluate clustering
def evaluate_clustering(pca_features, cluster_labels):
    if len(set(cluster_labels)) > 1:
        db_index = round(davies_bouldin_score(pca_features, cluster_labels), 3)
        s_score = round(silhouette_score(pca_features, cluster_labels), 3)
        ch_index = round(calinski_harabasz_score(pca_features, cluster_labels), 3)
        print(f"Davies-Bouldin Index: {db_index}")
        print(f"Silhouette Score: {s_score}")
        print(f"Calinski Harabasz Index: {ch_index}")
        return db_index, s_score, ch_index
    else:
        print("Silhouette score is not defined for a single cluster")
        return None, None, None

# Calculate evaluation metrics for the best DBSCAN clustering
db_dbscan, ss_dbscan, ch_dbscan = evaluate_clustering(pca_features, best_labels)

# Cluster profiling
if 'Cluster' in original_data.columns:
    cluster_profile = original_data.groupby('Cluster').mean()
    print(cluster_profile)

# Scatter plot of the PCA-reduced features with centroids and cluster names
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=best_labels, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering Results')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Count the number of points in each cluster
if 'Cluster' in original_data.columns:
    cluster_counts = original_data['Cluster'].value_counts().sort_index()
    print('Number of points in each cluster:')
    print(cluster_counts)

    # Plotting the number of data points in each cluster with annotations
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Data Points')
    plt.title('Number of Data Points in Each Cluster')

    # Annotate the number of points on the bars
    for index, value in enumerate(cluster_counts.values):
        barplot.text(index, value, str(value), color='black', ha="center")

    plt.show()

# Descriptive statistics for selected columns in each cluster without scaling
selected_columns = ['Credit', 'Age', 'Status', 'Education', 'Avg_Previous', 'Avg_Repayment', 'Avg_Amount']
selected_columns_2 = ['Credit', 'Avg_Previous', 'Avg_Amount']
if 'Cluster' in original_data.columns:
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

if 'Cluster' in original_data.columns:
    sns.pairplot(data=original_data[selected_columns_for_pairplot + ['Cluster']], hue="Cluster", palette='Set1')
    plt.show()

# Bar plot for means
if 'Cluster' in original_data.columns:
    means = original_data.groupby('Cluster')[selected_columns_2].mean().round(2)
    means.plot(kind='bar', figsize=(14, 7))
    plt.title('Means of Selected Features by Cluster')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=0)
    plt.show()

# Box plot for distribution
plt.figure(figsize=(20, 15))
for i, column in enumerate(selected_columns, 1):
    plt.subplot(3, 3, i)
    if 'Cluster' in original_data.columns:
        sns.boxplot(x='Cluster', y=column, data=original_data)
        plt.title(f'Boxplot of {column} by Cluster')
plt.tight_layout()
plt.show()
