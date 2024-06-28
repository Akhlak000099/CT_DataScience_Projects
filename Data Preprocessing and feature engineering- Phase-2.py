#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data Preprocessing
# Loading Data

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc

# *****Load dataset*****
data = pd.read_csv('dataset.csv')

# Display first few rows
print(data.head())


# *****Handling Missing Values*****
# Check for missing values
print(data.isnull().sum())

# Impute missing values (example using mean)
imputer = SimpleImputer(strategy='mean')
data['column_with_missing_values'] = imputer.fit_transform(data[['column_with_missing_values']])

# *****Feature Scaling*****
# Standardization (using StandardScaler)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['feature1', 'feature2']])

# *****Feature Engineering*****
# Feature Extraction (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# *****Create a DataFrame with principal components*****
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# *****Feature Encoding (One-Hot Encoding)*****
# Convert categorical variable into dummy/indicator variables (one-hot encoding)
data_encoded = pd.get_dummies(data, columns=['categorical_feature'])

# *****Dimensionality Reduction (t-SNE)*****
tsne = TSNE(n_components=2, random_state=0)
tsne_result = tsne.fit_transform(data_scaled)

# *****Plot t-SNE results*****
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data['target_variable'], cmap='viridis')
plt.colorbar()
plt.show()

# *****Model Evaluation*****
X_train, X_test, y_train, y_test = train_test_split(data.drop('target_variable', axis=1), 
                                                    data['target_variable'], test_size=0.2, random_state=0)

# *****Cross-validation*****
clf = DecisionTreeClassifier()

# *****Perform 5-fold cross-validation*****
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# *****Hyperparameter Tuning (Grid Search)*****
param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# *****Model Evaluation Metrics*****
y_pred = grid_search.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# *****ROC Curve and AUC Score*****
y_prob = grid_search.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("AUC Score:", auc(fpr, tpr))

# *****Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared*****
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

