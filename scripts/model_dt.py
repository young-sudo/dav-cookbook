#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
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

    # DecisionTreeClassifier
    dt_params = {
        'max_depth': range(1, 21),
        'max_features': ['sqrt', 'log2', None] # 'auto' is deprecated
    }

    dt_clf = DecisionTreeClassifier(random_state=42)
    grid_search_dt = GridSearchCV(dt_clf, dt_params, cv=5, scoring='accuracy')
    grid_search_dt.fit(X_train, y_train)

    # Grid search results, sorted by accuracy
    print(gridsearch2table(grid_search_dt))

    # Best Decision Tree
    best_dt = grid_search_dt.best_estimator_
    y_val_pred_dt = best_dt.predict(X_val)
    dt_accuracy = accuracy_score(y_val, y_val_pred_dt)

    print("Best Decision Tree Parameters:", grid_search_dt.best_params_)
    print("Decision Tree Validation Accuracy:", dt_accuracy)
    print(classification_report(y_val, y_val_pred_dt))

    # Visualize the best Decision Tree
    plt.figure(figsize=(50,25))
    plot_tree(best_dt, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
    # plt.show()
    plot_path = os.path.join(path_to_plots, "best_dt.png")
    plt.savefig(plot_path)

