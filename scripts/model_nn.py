#!/usr/bin/env python3

from sklearn.neighbors import KNeighborsClassifier
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

    # Nearest Neighbors
    knn_params = {
        'n_neighbors': range(15, 36),
        'weights': ['uniform', 'distance']
    }
    knn_clf = KNeighborsClassifier()
    grid_search_knn = run_grid_search(knn_clf, knn_params, X_train, y_train)

    # Best KNN model
    best_knn = grid_search_knn.best_estimator_
    y_val_pred_knn = best_knn.predict(X_val)
    knn_accuracy = accuracy_score(y_val, y_val_pred_knn)
    knn_report = classification_report(y_val, y_val_pred_knn, output_dict=True)

    print("Best KNN Parameters:", grid_search_knn.best_params_)
    print("KNN Validation Accuracy:", knn_accuracy)
    print(classification_report(y_val, y_val_pred_knn))

    print(gridsearch2table(grid_search_knn))

