#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:40:31 2024

@author: t.munkhsaruul
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load your dataset
os.chdir("/Users/sarthaksaxena/Documents/Sarthak/Readings/Masters/Sem 2/Machine Learning/Project/")
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
    return data, pca_features, kmeans

# Step 3: Determine the optimal number of clusters using silhouette analysis
def find_optimal_clusters_silhouette(scaler, data, columns_to_scale):
    silhouette_scores = []
    cluster_range = range(2, 11)  # Adjust the range based on your preference

    for n_clusters in cluster_range:
        _, pca_features, kmeans = scale_and_cluster(scaler, data, columns_to_scale, n_clusters)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(pca_features, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # Plot Silhouette Scores to choose the optimal number of clusters
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Number of Clusters (RobustScaler)')
    plt.show()
    
    # Return the cluster range and silhouette scores for further analysis
    return cluster_range, silhouette_scores

# Call the silhouette score function for RobustScaler
robust_cluster_range, robust_silhouette_scores = find_optimal_clusters_silhouette(RobustScaler(), data, columns_to_scale)

# Determine the optimal number of clusters based on silhouette score
optimal_clusters = robust_cluster_range[np.argmax(robust_silhouette_scores)]

# Perform clustering with the optimal number of clusters using RobustScaler
original_data, pca_features, kmeans = scale_and_cluster(RobustScaler(), data, columns_to_scale, optimal_clusters)

# Evaluate clustering
def evaluate_clustering(pca_features, cluster_labels):
    db_index = round(davies_bouldin_score(pca_features, cluster_labels), 3)
    s_score = round(silhouette_score(pca_features, cluster_labels), 3)
    ch_index = round(calinski_harabasz_score(pca_features, cluster_labels), 3)
    print(f"Davies-Bouldin Index: {db_index}")
    print(f"Silhouette Score: {s_score}")
    print(f"Calinski Harabasz Index: {ch_index}")
    return db_index, s_score, ch_index

db_kmeans, ss_kmeans, ch_kmeans = evaluate_clustering(pca_features, kmeans.labels_)

# Cluster profiling
cluster_profile = original_data.groupby('Cluster').mean()
print(cluster_profile)

# Scatter plot of the PCA-reduced features with centroids and cluster names
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title('KMeans Clustering Results')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(scatter, label='Cluster')

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='r', label='Centroids')

plt.legend()
plt.show()

# Count the number of points in each cluster
cluster_counts = original_data['Cluster'].value_counts().sort_index()
print('Number of points in each cluster:')
print(cluster_counts)

# Plotting the number of data points in each cluster with annotations
plt.figure(figsize=(6, 6))
barplot = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=['purple', 'yellow'])
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Number of Data Points in Each Cluster')

# Annotate the number of points on the bars
for index, value in enumerate(cluster_counts.values):
    barplot.text(index, value, str(value), color='black', ha="center")

plt.show()

# Descriptive statistics for selected columns in each cluster without scaling
selected_columns = ['Credit', 'Age', 'Status', 'Education', 'Avg_Previous', 'Mode Repayment Status', 'Avg_Amount']
selected_columns_2 = ['Credit', 'Avg_Previous', 'Avg_Amount']
selected_columns_3 = ['Credit', 'Age','Avg_Previous', 'Avg_Amount']
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
selected_columns_for_pairplot = ['Credit', 'Age', 'Status', 'Education', 'Avg_Previous', 'Mode Repayment Status', 'Avg_Amount']

sns.pairplot(data=original_data[selected_columns_for_pairplot + ['Cluster']], hue="Cluster", palette='Set1')
plt.show()

# Bar plot for means
means = original_data.groupby('Cluster')[selected_columns_2].mean().round(2)
means.plot(kind='bar', figsize=(14, 7))
plt.title('Means of Selected Features by Cluster')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.show()

# Box plot for distribution
plt.figure(figsize=(20, 15))
for i, column in enumerate(selected_columns_3, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Cluster', y=column, data=original_data)
    plt.title(f'Boxplot of {column} by Cluster')
plt.tight_layout()
plt.show()

# Visualize cluster centers with bar plot
plt.figure(figsize=(20, 8))
cluster_profile.plot(kind='bar')
plt.title('Cluster Centers of Numerical Features')
plt.xlabel('Cluster')
plt.xticks(rotation=0)
plt.ylabel('Feature Value')
plt.legend(loc='upper right')
plt.show()

# Pie charts for categorical distributions
def plot_pie_charts(data, attribute, clusters):
    cluster_values = data.groupby('Cluster')[attribute].value_counts(normalize=True).unstack().fillna(0)
    
    fig, axes = plt.subplots(1, len(clusters), figsize=(20, 5), subplot_kw=dict(aspect="equal"))
    fig.suptitle(f'Distribution of {attribute} per Cluster', fontsize=16)
    
    for i, cluster in enumerate(clusters):
        cluster_data = cluster_values.loc[cluster]
        axes[i].pie(cluster_data, labels=cluster_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2', len(cluster_data)))
        axes[i].set_title(f'Cluster {cluster}')
    
    plt.tight_layout()
    plt.show()

# Get unique clusters
clusters = original_data['Cluster'].unique()

# Plot pie charts for each categorical variable
plot_pie_charts(original_data, 'Credit', clusters)
plot_pie_charts(original_data, 'Age', clusters)
plot_pie_charts(original_data, 'Status', clusters)
plot_pie_charts(original_data, 'Education', clusters)
plot_pie_charts(original_data, 'Mode Repayment Status', clusters)