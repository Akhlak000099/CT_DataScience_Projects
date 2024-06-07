#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Map target names to species
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
iris_df['species'] = iris_df['species'].map(species_map)

# Bar plot showing the count of each species using pandas
plt.figure(figsize=(8, 5))
iris_df['species'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Count of each species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Scatter plot of sepal length vs. sepal width using matplotlib
plt.figure(figsize=(8, 5))
for species, color in zip(iris_df['species'].unique(), ['blue', 'red', 'green']):
    subset = iris_df[iris_df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=species, color=color)
plt.title('Sepal Length vs. Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.grid(True)
plt.show()

# Box plot showing the distribution of petal lengths for each species
plt.figure(figsize=(10, 6))
for species, color in zip(iris_df['species'].unique(), ['blue', 'red', 'green']):
    subset = iris_df[iris_df['species'] == species]
    plt.boxplot(subset['petal length (cm)'], positions=[species], widths=0.5, patch_artist=True, boxprops=dict(facecolor=color))
plt.title('Distribution of Petal Lengths')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.xticks([0, 1, 2], ['Setosa', 'Versicolor', 'Virginica'])
plt.grid(True)
plt.show()

# Histogram of petal width using NumPy and Matplotlib
plt.figure(figsize=(8, 5))
for species, color in zip(iris_df['species'].unique(), ['blue', 'red', 'green']):
    subset = iris_df[iris_df['species'] == species]
    plt.hist(subset['petal width (cm)'], bins=np.arange(0, 3, 0.25), label=species, alpha=0.7, color=color)
plt.title('Histogram of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




