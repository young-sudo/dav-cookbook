#!/usr/bin/env python3

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import os

from my_utils import gridsearch2table, run_grid_search

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

    # Support Vector Machine (RBF)
    svm_params = {
        'svc__C': [1, 10, 100, 500, 1000],
        'svc__gamma': [1, 0.1, 0.01]
    }
    svm_clf = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', random_state=42))])
    grid_search_svm = run_grid_search(svm_clf, svm_params, X_train, y_train)

    # Best SVM model
    best_svm = grid_search_svm.best_estimator_
    y_val_pred_svm = best_svm.predict(X_val)
    svm_accuracy = accuracy_score(y_val, y_val_pred_svm)
    svm_report = classification_report(y_val, y_val_pred_svm, output_dict=True)

    print("Best SVM Parameters:", grid_search_svm.best_params_)
    print("SVM Validation Accuracy:", svm_accuracy)
    print(classification_report(y_val, y_val_pred_svm))

    print(gridsearch2table(grid_search_svm))

