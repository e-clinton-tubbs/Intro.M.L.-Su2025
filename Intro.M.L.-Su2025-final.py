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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#DATA IMPORT
df = pd.read_csv("C:/Users/eclin/Documents/GitHub/Intro.M.L.-Su2025/Resource Use Classfication.csv")
np.random.seed(1234)
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

#Remove duplicates // winnie
columns_to_check = df.columns[2:]
df = df.drop_duplicates(subset=columns_to_check).reset_index(drop=True)

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
# 2.9.1 normalization
# produce a histogram to show the right skewness of the data
# Select and clean your column
col = 'Total Energy Use To Revenues USD in million'
data = df[col].dropna()

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Energy Use To Revenues')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2.9.2 loglp to compress right skewness
df['teur_log'] = np.log1p(df[col])
# 2.9.3 min/max scale log-values into [0,1]
scaler = MinMaxScaler()
df['teur_bin'] = scaler.fit_transform(df[['teur_log']])

#df['teur_bin'] = (df[col] > df[col].mean()).astype(int)

# produce a new histogram to show that the right skewness resolved
# produce a histogram to show the right skewness of the data
# Select and clean your column
col = 'teur_bin'
data = df[col].dropna()

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Energy Use To Revenues Recoded')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2.9 Environmental Materials Sourcing (FALSE(0)/TRUE(1)) - bool
df['ems_bin'] = df['Environmental Materials Sourcing'].map({False: 0, True: 1})

# 2.10 Renewable Energy Use (FALSE(0)/TRUE(1)) - bool
df['reu_bin'] = df['Renewable Energy Use'].map({False: 0, True: 1})
 
# 2.11 Green Buildings (FALSE(0)/TRUE(1)) - object
df['gb_bin'] = df['Green Buildings'].map({'FALSE': 0, 'TRUE': 1, 'no value': np.nan})

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

print(df.dtypes)

# see missing values
print(df.isnull().sum())

#SUPPOSED TO BE dropping rows w/t missing values //ECT
df = df.dropna(how='any',axis=0)

# Check that missing values have been dropped
print(df.isnull().sum())

#    3. fit the neural network

#dropping index column
df = df.drop(columns=['Unnamed: 0'])

# Split the data into features (X) and target (y)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Split

# 0) Print out your actual column names so you can see exactly what Pandas sees:
#print(df.columns.tolist())

# 1) drop rows that are missing the features you care about
to_keep = [
  'teur_bin',
  'pwe_bin','pee_bin','pesc_bin','rrt_bin',
  'twe_bin','tee_bin','emt_bin','ems_bin',
  'reu_bin','gb_bin','escm_bin'
]
df = df.dropna(subset=to_keep)

# 2) pick X & y
X = df.drop(columns='rus_bin')
y = df['rus_bin']

# 3) keep only numeric features
X = X.select_dtypes(include=['number'])

# 1) Stratified split → X_tr is a DataFrame
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Recombine and sample on pandas objects
import pandas as pd
train = pd.concat([X_train, y_train.rename('rus_bin')], axis=1)

# Separate classes
maj = train[train.rus_bin == 1]
min = train[train.rus_bin == 0]

# Upsample minority
min_up = min.sample(n=len(maj), replace=True, random_state=42)

# Put back together
train_bal = pd.concat([maj, min_up], axis=0)

# Split back out
X_train_bal = train_bal.drop(columns='rus_bin')
y_train_bal = train_bal['rus_bin']

print("X_train.shape:", X_train_bal.shape)
print("y_train.shape:", y_train_bal.shape)

#checking how many of 0 & 1 are in the rus_bin
unique, counts = np.unique(y_train_bal, return_counts=True)
print(dict(zip(unique,counts)))

#print(df['Resource Use Score'].value_counts())

print("Y original train class counts:", y_train.value_counts())
print("Y After balancing:", y_train_bal.value_counts())

print("X Original train class counts:", X_train.value_counts())
print("X After balancing:", X_train_bal.value_counts())

# 3) # scaling data
scaler = StandardScaler().fit(X_train_bal)
#scaler = StandardScaler()
X_train_bal = scaler.transform(X_train_bal)
X_test = scaler.transform(X_test)

# RUN THE LOGI. REG. HERE (yoinked wholesale from Julian's code)

logreg = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
#logreg_result=logreg.fit(X, y)
logreg_result=logreg.fit(X, y)
y_pred_logreg=logreg.predict(X)

print(f'Accuracy: {accuracy_score(y, y_pred_logreg)}')
print(f'Confusion Matrix: {confusion_matrix(y, y_pred_logreg)}')
print(f'Classification Report: {classification_report(y, y_pred_logreg)}')

print(logreg.predict_proba(X)) # Predicted Output False/True 
print(logreg.predict(X)) #Actual Predictions

# Cofusion matrix as heat map 
cm = confusion_matrix(y, y_pred_logreg)
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted False', 'Predicted True'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual False', 'Actual Ture'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title('Confusion matrix for Logistic Regression for Resource Use Data')
plt.show()


from sklearn.metrics import roc_curve, roc_auc_score

logreg_roc_auc = roc_auc_score(y, logreg.predict(X))
fpr, tpr, thresholds = roc_curve(y, logreg.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logreg_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

print(
      "Classes: ", logreg.classes_, "\n",
      "Intercept: ", logreg.intercept_,"\n",
      "Coefficients: ", logreg.coef_
      )

# Instantiate and train the neural network with specified parameters
clf = MLPClassifier(random_state=1, 
                    hidden_layer_sizes=(120, 80, 40),  #  3 hidden layers with 120, 80, 40 neurons
                    max_iter=400,  
                    activation = "relu",  # Sigmoid activation function
                    solver = "sgd",  # Adam optimizer (stochastic gradient descent method)
                    learning_rate="constant",
                    learning_rate_init=0.1,
                    alpha=0.0001,  # Regularization term
                    early_stopping=True).fit(X_train_bal, y_train_bal)

# Predict on the test set
y_test_pred = clf.predict(X_test)
# Calculate predicted probabilities (for further analysis if needed)
clf.predict_proba(X_test)

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

layer_coefs = clf.coefs_
layer_intercepts = clf.intercepts_
iterations = clf.n_iter_

# Plot the loss curve to visualize loss decay over iterations
plt.plot(clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Perform hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)], 
    'max_iter': [1, 100, 200, 300, 400, 500],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05, 1],
    'learning_rate': ['constant','adaptive'],
    'learning_rate_init': [0.1, 1],
}

# Grid search with cross-validation
grid = GridSearchCV(clf, param_grid, n_jobs= -1, cv=5, verbose=2)
grid.fit(X_train_bal, y_train_bal)

# Print the best hyperparameters
print(grid.best_params_) 

# Predict using the best model from GridSearchCV
grid_predictions = grid.predict(X_test) 

# Print the accuracy of the grid search model
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))

#    4. model diagnostic

#4.1.error propagation

import numpy as np

class BackpropMLP:
    def __init__(self, n_inputs, n_hidden=9, lr=0.001, seed=42):
        rng = np.random.RandomState(seed)
        # 1) randomly initialize weights & thresholds (biases) from (0,1)
        self.W1 = rng.rand(n_inputs, n_hidden)    # input → hidden weights
        self.b1 = rng.rand(n_hidden)              # hidden thresholds
        self.W2 = rng.rand(n_hidden, 1)           # hidden → output weights
        self.b2 = rng.rand(1)                     # output threshold
        self.lr = lr

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _dsigmoid(self, a):
        # derivative of sigmoid at activation a
        return a * (1.0 - a)

    def fit(self, X, y, max_epochs=1000, tol=1e-6):
        """
        X: (n_samples, n_inputs), y: (n_samples,) with values 0 or 1
        """
        n_samples = X.shape[0]
        for epoch in range(max_epochs):
            epoch_loss = 0.0

            # 2) REPEAT until termination
            for xi, yi in zip(X, y):
                # 4) FORWARD: compute activations
                z1 = xi.dot(self.W1) + self.b1           # hidden pre-act
                a1 = self._sigmoid(z1)                   # hidden output
                z2 = a1.dot(self.W2) + self.b2           # output pre-act
                a2 = self._sigmoid(z2).ravel()           # final output

                # accumulate mean‐squared error
                epoch_loss += 0.5 * (yi - a2)**2

                # 5) DELTA for output neuron: δ² = y(1−y)(t−y)
                delta2 = (yi - a2) * self._dsigmoid(a2)   # shape (1,)

                # 6) DELTA for hidden neurons: δ¹ = h(1−h) * (W2 · δ²)
                delta1 = self._dsigmoid(a1) * (self.W2.ravel() * delta2)

                # 7) UPDATE weights & thresholds
                # hidden→output
                self.W2 += self.lr * np.outer(a1, delta2)
                self.b2 += self.lr * delta2

                # input→hidden
                self.W1 += self.lr * np.outer(xi, delta1)
                self.b1 += self.lr * delta1

            # average loss this epoch
            epoch_loss /= n_samples

            # 9) TERMINATION check
            if epoch_loss < tol:
                print(f'Converged at epoch {epoch} with loss {epoch_loss:.6e}')
                break
        else:
            print(f'Max epochs reached, final loss: {epoch_loss:.6e}')

    def predict_proba(self, X):
        a1 = self._sigmoid(X.dot(self.W1) + self.b1)
        a2 = self._sigmoid(a1.dot(self.W2) + self.b2)
        return a2.ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
#    5. plot it
#    6. profit
#END
