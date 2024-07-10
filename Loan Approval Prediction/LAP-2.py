#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Model Training and Evaluation

#  Splitting Data into Train and Test Sets
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop('loan_status', axis=1)  # Assuming 'loan_status' is the target variable
y = df['loan_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Training and Evaluation
# Importing all the models
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize models
kmeans = KMeans(n_clusters=2, random_state=42)
hierarchical = AgglomerativeClustering(n_clusters=2)
dbscan = DBSCAN(eps=0.5, min_samples=5)
gmm = GaussianMixture(n_components=2, random_state=42)

random_forest = RandomForestClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)
bagging = BaggingClassifier(random_state=42)
adaboost = AdaBoostClassifier(random_state=42)
voting = VotingClassifier(estimators=[('rf', random_forest), ('gb', gradient_boosting)], voting='soft')

logistic_regression_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
logistic_regression_l2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
sgd_classifier_l1 = SGDClassifier(loss='log', penalty='l1', random_state=42)
sgd_classifier_l2 = SGDClassifier(loss='log', penalty='l2', random_state=42)
sgd_classifier_elasticnet = SGDClassifier(loss='log', penalty='elasticnet', random_state=42, l1_ratio=0.15)

# List of models
models = {
    'KMeans': kmeans,
    'Hierarchical': hierarchical,
    'DBSCAN': dbscan,
    'GaussianMixture': gmm,
    'RandomForest': random_forest,
    'GradientBoosting': gradient_boosting,
    'Bagging': bagging,
    'AdaBoost': adaboost,
    'Voting': voting,
    'LogisticRegression L1': logistic_regression_l1,
    'LogisticRegression L2': logistic_regression_l2,
    'SGDClassifier L1': sgd_classifier_l1,
    'SGDClassifier L2': sgd_classifier_l2,
    'SGDClassifier ElasticNet': sgd_classifier_elasticnet
}

# Function to train and evaluate each model
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Train and evaluate each model
for name, model in models.items():
    print(f"Training and evaluating: {name}")
    accuracy, report = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print("------------------------------------------")

    

