#PREAMBLE
# NAME: Intro.M.L. Group Project
# DATE: 06/11/2025
# CLASS: Introduction to Machine Learning Su2025
# WHOSE: Simin.tahmasebi.gandomkari; chujie.wang; winnie.tan; eric.tubbs

#MODS. IMPORT
import numpy as np #chujie.wang suggestion
import pandas as pd
import os
import matplotlib.pyplot as plt

#DATA IMPORT
df = pd.read_csv("C:/Users/eclin/Documents/GitHub/Intro.M.L.-Su2025/Resource Use Classfication.csv")

#WORKSPACE
#    1. exploratory data analysis & data cleaning
 
print(df.info())     # column dtypes & non-null counts //COPILOT
print(df.head())      # peek at first rows  //COPILOT
print(df.describe())  # stats on numeric cols  //COPILOT

#    2. data preprocessing

# Check for missing values //from S.S.S.T.
print(df.isnull().sum())

#ECT; 114 out of 422 firms have no "total energy use to revenues USD in millions"
#ECT; otherwise looks good?

df = df.dropna(how='any',axis=0)  #SUPPOSED TO BE dropping rows w/t missing values //ECT

# Check that missing values have been dropped //from S.S.S.T.
print(df.isnull().sum())

# check for class imbalance
print(df['Resource Use Score'].value_counts())

# Summary stats of data
print(df.iloc[:,1:].describe())  # only summary stats for numeric columns

# var. hardcoding loop 
# 2.1 resource use score (bad(0)/good(1))
# assume df['level'] contains strings "good" & "bad"
df['level_bin'] = df['Resource Use Score'].map({'bad': 0, 'good': 1})
# if any other values, they’ll become NaN – you can .fillna(…) or check .isna()

# 2.2 Policy Water Efficiency (FALSE(0)/TRUE(1))

# 2.3 Policy Energy Efficiency (FALSE(0)/TRUE(1))

# 2.4 Policy Environmental Supply Chain (FALSE(0)/TRUE(1))

# 2.5 Resource Reduction Targets (FALSE(0)/TRUE(1))

# 2.6 Targets Water Efficiency (FALSE(0)/TRUE(1))

# 2.7 Targets Energy Efficiency (FALSE(0)/TRUE(1))

# 2.8 Environment Management Training (FALSE(0)/TRUE(1))

# 2.9 Total Energy Use To Revenues USD in million
# normalization

# 2.9 Environmental Materials Sourcing (FALSE(0)/TRUE(1))

# 2.10 Renewable Energy Use (FALSE(0)/TRUE(1))

# 2.11 Green Buildings (FALSE(0)/TRUE(1))

# 2.12 Environmental Supply Chain Management

#    3. fit the neural network

# Split the data into features (X) and target (y)
from sklearn.model_selection import train_test_split

#X = df.drop('deposit_new', axis=1)
#y = df['deposit_new']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#0
# %%
# scaling data
from sklearn.preprocessing import StandardScaler


#    4. model diagnostic
#    5. plot it
#    6. profit
#END