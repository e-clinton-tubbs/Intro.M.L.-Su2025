#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:15:58 2024

@author: t.munkhsaruul
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load and Explore the Dataset
os.chdir("/Users/sarthaksaxena/Documents/Sarthak/Readings/Masters/Sem 2/Machine Learning/Project/")
data = pd.read_csv('cleaned_data_clustering.csv')

# Display the first few rows and check for any missing values
print(data.head())
print(data.info())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Create average columns
data['Avg_Previous'] = data[['Previous 1', 'Previous 2', 'Previous 3', 'Previous 4', 'Previous 5', 'Previous 6']].mean(axis=1)
data['Avg_Repayment'] = data[['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']].mean(axis=1)
data['Avg_Amount'] = data[['Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6']].mean(axis=1)
repayment_columns = ['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']
data['Mode Repayment Status'] = data[repayment_columns].mode(axis=1)[0]

# Ensure 'Sex' is treated as a categorical variable
data['Sex'] = data['Sex'].astype(str)

# Define Columns to Scale
columns_to_scale = ['Avg_Amount', 'Avg_Previous']

scaler = RobustScaler()
scaled_features = scaler.fit_transform(data[columns_to_scale])
data[columns_to_scale] = scaled_features

# Select features for clustering
X = data[columns_to_scale]

# Fit K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Evaluate silhouette score
silhouette_avg = silhouette_score(X, data['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Visualize clusters
sns.scatterplot(x='Avg_Amount', y='Avg_Previous', hue='Cluster', data=data, palette='Set1', legend='full')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', marker='X')
plt.title('K-means Clustering of Bank Clients with Avg_Amount and Avg_Previous')
plt.xlabel('Avg_Amount')
plt.ylabel('Avg_Previous')
plt.legend()
plt.show()

# Count the number of points in each cluster
cluster_counts = data['Cluster'].value_counts().sort_index()
print('Number of points in each cluster:')
print(cluster_counts)

# Plotting the number of data points in each cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Number of Data Points in Each Cluster')
plt.show()

# Descriptive statistics for each cluster
selected_columns = ['Credit', 'Mode Repayment Status', 'Age', 'Status', 'Education', 'Avg_Previous', 'Avg_Amount', 'Avg_Repayment']
cluster_descriptive_stats = data.groupby('Cluster')[selected_columns].describe().transpose()

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

# Plotting descriptive statistics
# Bar plot for means
means = data.groupby('Cluster')[['Avg_Previous', 'Avg_Amount']].mean().round(2)
means.plot(kind='bar', figsize=(18, 10))
plt.title('Means of Selected Features by Cluster')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.show()

# Bar plot for Credit
means = data.groupby('Cluster')[['Credit']].mean().round(2)
means.plot(kind='bar', figsize=(18, 7))
plt.title('Means of Credit by Cluster')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.show()

# Box plot for distribution
plt.figure(figsize=(20, 15))
for i, column in enumerate(selected_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Cluster', y=column, data=data)
    plt.title(f'Boxplot of {column} by Cluster')
plt.tight_layout()
plt.show()

# Pie chart for Sex distribution by cluster
num_clusters = data['Cluster'].nunique()
fig, axes = plt.subplots(1, num_clusters, figsize=(15, 6))

for i, cluster_id in enumerate(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_id]['Sex'].value_counts()
    axes[i].pie(cluster_data, labels=cluster_data.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Cluster {cluster_id}')

fig.suptitle('Sex Distribution by Cluster')
plt.tight_layout()
plt.show()


# Pie chart for Education distribution by cluster
num_clusters = data['Cluster'].nunique()
fig, axes = plt.subplots(1, num_clusters, figsize=(20, 8))

for i, cluster_id in enumerate(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_id]['Education'].value_counts()
    axes[i].pie(cluster_data, labels=cluster_data.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Cluster {cluster_id}')

fig.suptitle('Education Distribution by Cluster')
plt.tight_layout()
plt.show()

# Bar plots for education levels within each cluster
num_clusters = data['Cluster'].nunique()
fig, axes = plt.subplots(1, num_clusters, figsize=(20, 6), sharey=True)

for i, cluster_id in enumerate(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_id]['Education'].value_counts()
    sns.barplot(x=cluster_data.index, y=cluster_data.values, ax=axes[i])
    axes[i].set_title(f'Cluster {cluster_id}')
    axes[i].set_xlabel('Education Level')
    axes[i].set_ylabel('Count')
 # Add labels on the bars
    for j in range(len(cluster_data)):
        axes[i].text(j, cluster_data.values[j], str(cluster_data.values[j]), ha='center', va='bottom')
        
# Pie chart for Status distribution by cluster
num_clusters = data['Cluster'].nunique()
fig, axes = plt.subplots(1, num_clusters, figsize=(20, 8))

for i, cluster_id in enumerate(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_id]['Status'].value_counts()
    axes[i].pie(cluster_data, labels=cluster_data.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Cluster {cluster_id}')

fig.suptitle('Status Distribution by Cluster')
plt.tight_layout()
plt.show()

# Bar plots for Mode of Repayment Status within each cluster
num_clusters = data['Cluster'].nunique()
fig, axes = plt.subplots(1, num_clusters, figsize=(20, 6), sharey=True)

for i, cluster_id in enumerate(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_id]['Mode Repayment Status'].value_counts()
    sns.barplot(x=cluster_data.index, y=cluster_data.values, ax=axes[i])
    axes[i].set_title(f'Cluster {cluster_id}')
    axes[i].set_xlabel('Mode of Repayment Status')
    axes[i].set_ylabel('Count')
# Add labels on the bars
    for j in range(len(cluster_data)):
        axes[i].text(j, cluster_data.values[j], str(cluster_data.values[j]), ha='center', va='bottom')
        
# Pie chart for Status distribution by cluster
num_clusters = data['Cluster'].nunique()
fig, axes = plt.subplots(1, num_clusters, figsize=(20, 8))

for i, cluster_id in enumerate(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_id]['Mode Repayment Status'].value_counts()
    axes[i].pie(cluster_data, labels=cluster_data.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Cluster {cluster_id}')

fig.suptitle('Status Distribution by Mode Repayment Status')
plt.tight_layout()
plt.show()

# Count plot of Education level by cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='Education', hue='Cluster', data=data)
plt.title('Count of Education Levels by Cluster')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Cluster')
plt.show()

data_groupby7 = data.groupby('Education')[['Cluster']].mean()

ax = data_groupby7.plot.bar(figsize=(8, 6), width=0.7)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.show()

sns.displot(data = data, x="Cluster", col='Education',multiple="dodge", stat='density', shrink=0.8)
plt.suptitle("Frequency plots for the variable 'Sex', by cluster",y=1.05)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
