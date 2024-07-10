#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#  Data Preprocessing and Feature Engineering

# Loading the Data
import pandas as pd

# Load the dataset
df = pd.read_csv('loan_dataset.csv')

# Display the first few rows of the dataframe
print(df.head())

# Handling Missing Values
# Check for missing values
print(df.isnull().sum())

# Handle missing values (example: fill with mean or median)
df['income'].fillna(df['income'].median(), inplace=True)


# Encoding Categorical Variables
# Convert categorical variables into numerical using Label Encoding or One-Hot Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])


# Feature Scaling
# Scale numerical features (example: StandardScaler)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['income'] = scaler.fit_transform(df['income'].values.reshape(-1, 1))


# Feature Engineering
# Create new features or transform existing ones
df['loan_income_ratio'] = df['loan_amount'] / df['income']



