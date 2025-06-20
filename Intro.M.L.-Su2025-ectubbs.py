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

# var. hardcoding loop 
# 2.1 resource use score (bad(0)/good(1)) - object
# assume df['level'] contains strings "good" & "bad"
df['rus_bin'] = df['Resource Use Score'].map({'bad': 0, 'good': 1})
# if any other values, they’ll become NaN – you can .fillna(…) or check .isna()

# 2.2 Policy Water Efficiency (FALSE(0)/TRUE(1)) - bool
df['pwe_bin'] = df['Policy Water Efficiency'].map({False: 0, True: 1})

# 2.3 Policy Energy Efficiency (FALSE(0)/TRUE(1)) - bool
df['pee_bin'] = df['Policy Energy Efficiency'].map({False: 0, True: 1})

# 2.4 Policy Environmental Supply Chain (FALSE(0)/TRUE(1)) - bool 
df['pesc_bin'] = df['Policy Environmental Supply Chain'].map({False: 0, True: 1})

# 2.5 Resource Reduction Targets (FALSE(0)/TRUE(1)) - bool
df['rrt_bin'] = df['Resource Reduction Targets'].map({False: 0, True: 1})

# 2.6 Targets Water Efficiency (FALSE(0)/TRUE(1)) - bool
df['twe_bin'] = df['Targets Water Efficiency'].map({False: 0, True: 1})

# 2.7 Targets Energy Efficiency (FALSE(0)/TRUE(1)) - bool 
df['tee_bin'] = df['Targets Energy Efficiency'].map({False: 0, True: 1})

# 2.8 Environment Management Training (FALSE(0)/TRUE(1)) - bool
df['emt_bin'] = df['Environment Management Training'].map({False: 0, True: 1})

# 2.9 Total Energy Use To Revenues USD in million - float64
# normalization

# 2.9 Environmental Materials Sourcing (FALSE(0)/TRUE(1)) - bool
df['ems_bin'] = df['Environmental Materials Sourcing'].map({False: 0, True: 1})

# 2.10 Renewable Energy Use (FALSE(0)/TRUE(1)) - bool
df['reu_bin'] = df['Renewable Energy Use'].map({False: 0, True: 1})
 
# 2.11 Green Buildings (FALSE(0)/TRUE(1)) - object
#df['gb_bin'] = df['Green Buildings'].map({'FALSE': 0, 'TRUE': 1, 'no value': np.nan})

# 2.11.1 normalize casing & strip whitespace
df['GB_clean'] = df['Green Buildings'].str.strip().str.upper()

# 2.11.2 map strings to integers, let unmapped become <NA>
gb_map = {'TRUE': 1, 'FALSE': 0, 'NO VALUE': pd.NA}
df['gb_bin'] = df['GB_clean'].map(gb_map).astype('Int64')
print(df['gb_bin'].value_counts(dropna=False))

# 2.12 Environmental Supply Chain Management (FALSE(0)/TRUE(1)) - bool
df['escm_bin'] = df['Environmental Supply Chain Management'].map({False: 0, True: 1})

# check for class imbalance
print(df['Resource Use Score'].value_counts())

# Summary stats of data
print(df.iloc[:,1:].describe())  # only summary stats for numeric columns

#print(df.dtypes)

# see missing values
print(df.isnull().sum())

#SUPPOSED TO BE dropping rows w/t missing values //ECT
df = df.dropna(how='any',axis=0)

# Check that missing values have been dropped
print(df.isnull().sum())

#    3. fit the neural network

# %%
# Split the data into features (X) and target (y)
from sklearn.model_selection import train_test_split
# --- after df is preprocessed and missing rows dropped --
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# Split
X = df.drop(columns='rus_bin')
y = df['rus_bin']

# Ensure numeric
assert all(dtype.kind in 'biuf' for dtype in X.dtypes), "All X columns must be numeric"

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit MLP
clf = MLPClassifier(
    random_state=1,
    hidden_layer_sizes=(5,),
    max_iter=200,
    activation="logistic",
    solver="adam",
    learning_rate="constant",
    learning_rate_init=0.001,
    alpha=0.0001,
    early_stopping=True
)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#    4. model diagnostic
#    5. plot it
#    6. profit
#END
# %%
