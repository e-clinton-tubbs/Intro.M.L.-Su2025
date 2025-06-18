#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:35:48 2024

@author: sarthaksaxena
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial import Voronoi, voronoi_plot_2d
import itertools
import openpyxl as xl
from tabulate import tabulate
import matplotlib.pyplot as plt

# Load the dataset
os.chdir("/Users/sarthaksaxena/Documents/Sarthak/Readings/Masters/Sem 2/Machine Learning/Project/")
df = pd.read_csv('cleaned_data_clustering.csv')

# Summary of the dataset
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if any)
# df.fillna(method='ffill', inplace=True)

categorical_features = ['Sex', 'Education', 'Status', 'Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']

df['Sex'] = df['Sex'].replace({1: 'Male', 2: 'Female'})
df['Education'] = df['Education'].replace({1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others'})
df['Status'] = df['Status'].replace({1: 'Married', 2: 'Single', 3: 'Others'})
replacement_dict = {
    -2: 'no credit consumption',
    -1: 'repaid duly',
    0: 'use of revolving credit',
    1: 'payment delay of 1 month',
    2: 'payment delay of 2 months',
    3: 'payment delay of 3 months',
    4: 'payment delay of 4 months',
    5: 'payment delay of 5 months',
    6: 'payment delay of 6 months',
    7: 'payment delay of 7 months',
    8: 'payment delay of 8 months',
    9: 'payment delay of 9 or more months'}
df[['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']] = df[['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']].replace(replacement_dict)

#feature engineering
df['total_amount'] = df[['Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6']].sum(axis=1)
df['total_previous'] = df[['Previous 1', 'Previous 2', 'Previous 3', 'Previous 4', 'Previous 5', 'Previous 6']].sum(axis=1)
df['avg_amount'] = df[['Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6']].mean(axis=1)
df['avg_previous'] = df[['Previous 1', 'Previous 2', 'Previous 3', 'Previous 4', 'Previous 5', 'Previous 6']].mean(axis=1)
repayment_columns = ['Repayment 1', 'Repayment 2', 'Repayment 3', 'Repayment 4', 'Repayment 5', 'Repayment 6']
df['Mode_Repayment_Status'] = df[repayment_columns].mode(axis=1)[0]

continuous_features = ['Credit', 'Age', 'Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6', 'Previous 1', 'Previous 2', 'Previous 3', 'Previous 4', 'Previous 5', 'Previous 6', 'total_amount', 'total_previous', 'avg_amount', 'avg_previous']
selected_total_features = ['total_amount', 'total_previous']
selected_avg_features = ['avg_amount', 'avg_previous']

# Boxplot to check for outliers
plt.figure(figsize=(15, 10))
columns_to_include = ['Credit', 'Amount 1', 'Amount 2', 'Amount 3', 'Amount 4', 'Amount 5', 'Amount 6', 'Previous 1', 'Previous 2', 'Previous 3', 'Previous 4', 'Previous 5', 'Previous 6']
df_subset = df[columns_to_include]
sns.boxplot(data=df_subset)
plt.xticks(rotation=90)
plt.show()

# Visualizing Categorical Features
for feature in categorical_features:
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x=feature)
    plt.title(f'{feature} Distribution')
    plt.xticks(rotation=90)
    plt.legend(title=feature)
    plt.show()

# Visualizing Continuous Features
for feature in continuous_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()
    
    
# Visualizing Relationships Between Categorical and Continuous Features
for cont_feature in continuous_features:
    for cat_feature in categorical_features:
        plt.figure(figsize=(20, 12))
        means = df.groupby(cat_feature)[cont_feature].mean()
        plt.bar(means.index, means.values)
        plt.title(f'{cont_feature} by {cat_feature}')
        plt.xlabel(cat_feature)
        plt.ylabel(cont_feature)
        plt.xticks(rotation=90)
        plt.show()

# Visualizing Relationships Between Credit by education and sex with median.        
g = sns.FacetGrid(df, col="Sex", height=6, aspect=1)
g.map(sns.boxplot, "Education", "Credit")
g.add_legend()
plt.show()

#EDA Code end