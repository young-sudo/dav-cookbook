#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

from my_utils import gridsearch2table

path = os.getcwd()
parent = os.path.dirname(path) # extract parent dir in cross platform way

path_to_plots = os.path.join(parent, "plots")
path_to_data = os.path.join(parent, "data")

if __name__ == "__main__":
    train = pd.read_csv(os.path.join(path_to_data, "train_cleaned.csv"))

    # Separate features and target variable
    X = train.drop("Survived", axis=1)
    y = train["Survived"]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training in process...")

    # RandomForestClassifier
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': range(1, 21),
        'max_features': ['sqrt', 'log2', None] # 'auto' is deprecated
    }

    rf_clf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(rf_clf, rf_params, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)

    # Grid search results, sorted by accuracy
    print(gridsearch2table(grid_search_rf))

    # Best Random Forest
    best_rf = grid_search_rf.best_estimator_
    y_val_pred_rf = best_rf.predict(X_val)
    rf_accuracy = accuracy_score(y_val, y_val_pred_rf)

    print("Best Random Forest Parameters:", grid_search_rf.best_params_)
    print("Random Forest Validation Accuracy:", rf_accuracy)
    print(classification_report(y_val, y_val_pred_rf))
