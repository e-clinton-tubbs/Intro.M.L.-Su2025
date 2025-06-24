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

X = X.select_dtypes(include=['number'])
print(X.dtypes)  # Should now list only numeric dtypes


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Fit MLP
#clf = MLPClassifier(
#    random_state=1,
#    hidden_layer_sizes=(5,),
#    max_iter=200,
#    activation="logistic",
#    solver="adam",
#    learning_rate="constant",
#    learning_rate_init=0.001,
#    alpha=0.0001,
#    early_stopping=True
#)
#clf.fit(X_train_scaled, y_train)

# Evaluate
#y_pred = clf.predict(X_test_scaled)
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

# %%
# scaling data
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Instantiate and train the neural network with specified parameters
clf = MLPClassifier(random_state=1, 
                    hidden_layer_sizes=(5,),  # One hidden layer with 5 neurons
                    max_iter=200,  
                    activation = "logistic",  # Sigmoid activation function
                    solver = "adam",  # Adam optimizer (stochastic gradient descent method)
                    learning_rate="constant",
                    learning_rate_init=0.001,
                    alpha=0.0001,  # Regularization term
                    early_stopping=True).fit(X_train, y_train)

# Predict on the test set
y_test_pred = clf.predict(X_test)
# Calculate predicted probabilities (for further analysis if needed)
clf.predict_proba(X_test)

#%%
# Extract and print weights and biases of the trained neural network
print(f'Weights between the input and the hidden layer: {clf.coefs_[0]}')  
print(f'Weights between the hidden layer and the output: {clf.coefs_[1]}')

print(f'Value of w_0: {clf.coefs_[0][0][0]}')
print(f'Value of w_1: {clf.coefs_[0][1][0]}')

print(f'Bias values of first hidden layer: {clf.intercepts_[0]}')
print(f'Bias values of first hidden layer: {clf.intercepts_[1]}')

# Print model parameters and evaluation metrics
params = clf.get_params()

print(f'Accuracy: {accuracy_score(y_test, y_test_pred)}')
print(f'Confusion Matrix: {confusion_matrix(y_test, y_test_pred)}')
print(f'Classification Report: {classification_report(y_test, y_test_pred)}')

# Visualize the confusion matrix using a heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

#%%
layer_coefs = clf.coefs_
layer_intercepts = clf.intercepts_
iterations = clf.n_iter_

# Plot the loss curve to visualize loss decay over iterations
plt.plot(clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

#%%
# Perform hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)], 
    'max_iter': [200, 250, 300],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Grid search with cross-validation
grid = GridSearchCV(clf, param_grid, n_jobs= -1, cv=5, verbose=2)
grid.fit(X_train, y_train)

# Print the best hyperparameters
print(grid.best_params_) 

# Predict using the best model from GridSearchCV
grid_predictions = grid.predict(X_test) 

# Print the accuracy of the grid search model
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))



# %%


#    4. model diagnostic

#4.1.error propagation

import numpy as np

class SimpleMLP:
    def __init__(self, n_inputs, n_hidden=5, lr=0.001, seed=42):
        rng = np.random.RandomState(seed)
        # Step 1: initialize weights in (0,1)
        self.W1 = rng.rand(n_inputs, n_hidden)      # input → hidden
        self.b1 = rng.rand(n_hidden)                # hidden biases
        self.W2 = rng.rand(n_hidden, 1)             # hidden → output
        self.b2 = rng.rand(1)                       # output bias
        self.lr = lr

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _dsigmoid(self, a):
        # derivative of sigmoid given activation a
        return a * (1 - a)

    def fit(self, X, y, epochs=1000, tol=1e-4):
        """
        X: array-like, shape (n_samples, n_inputs)
        y: array-like, shape (n_samples,) with values 0 or 1
        """
        n_samples = X.shape[0]
        for epoch in range(epochs):
            loss = 0
            for xi, yi in zip(X, y):
                # --- forward pass (step 4) ---
                z1 = xi.dot(self.W1) + self.b1           # hidden linear
                a1 = self._sigmoid(z1)                   # hidden activation
                z2 = a1.dot(self.W2) + self.b2           # output linear
                a2 = self._sigmoid(z2).ravel()           # output activation

                # accumulate simple squared‐error loss
                loss += 0.5 * (yi - a2)**2

                # --- backward pass (step 5 & 6) ---
                # step 5: delta for output layer
                delta2 = (yi - a2) * self._dsigmoid(a2)   # shape (1,)

                # step 6: delta for hidden layer
                # sum over downstream weights * delta2, then * sigmoid’
                delta1 = self._dsigmoid(a1) * (self.W2.ravel() * delta2)

                # --- parameter updates (step 7) ---
                # hidden→output weights & output bias
                self.W2 += self.lr * np.outer(a1, delta2)   # (n_hidden,1)
                self.b2 += self.lr * delta2

                # input→hidden weights & hidden biases
                self.W1 += self.lr * np.outer(xi, delta1)    # (n_inputs,n_hidden)
                self.b1 += self.lr * delta1

            # average loss over dataset
            loss /= n_samples
            if loss < tol:
                print(f’Converged at epoch {epoch}, loss={loss:.6f}’)
                break

    def predict_proba(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1.dot(self.W2) + self.b2
        return self._sigmoid(z2).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# --- USAGE EXAMPLE ---
# after you’ve preprocessed and split your DataFrame:
# X_train, X_test are numpy arrays, y_train, y_test are 0/1 vectors

# Initialize & train
mlp = SimpleMLP(n_inputs=X_train.shape[1], n_hidden=5, lr=0.001)
mlp.fit(X_train, y_train, epochs=200)

# Evaluate
y_pred = mlp.predict(X_test)
acc = (y_pred == y_test).mean()
print("Test accuracy:", acc)


#    5. plot it
#    6. profit
#END
# %%
