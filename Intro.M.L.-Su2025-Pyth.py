#PREAMBLE
# NAME: Intro.M.L. Group Project
# DATE: 06/11/2025
# CLASS: Introduction to Machine Learning Su2025
# WHOSE: Simin.tahmasebi.gandomkari; chujie.wang; winnie.tan; eric.tubbs

#MODS. IMPORT
import pandas as pd
import os
import matplotlib.pyplot as plt

#DATA IMPORT
df = pd.read_csv("C:/Users/eclin/Documents/GitHub/Intro.M.L.-Su2025/Resource Use Classfication.csv")

#WORKSPACE
#    1. exploratory data analysis & data cleaning
 
df.info()      # column dtypes & non-null counts //COPILOT
df.head()      # peek at first rows  //COPILOT
df.describe()  # stats on numeric cols  //COPILOT

df = df.dropna(axis=0, thresh=int(0.6*len(df.columns))) #handling missing values by dropping them as N/A? //COPILOT


#    2. data preprocessing
#    3. fit the neural network
#    4. model diagnostic
#    5. plot it
#    6. profit
#END