#!/usr/bin/env python3

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore") # suppress failed-to-converge warnings

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

    mlp_params = {
        'mlp__hidden_layer_sizes': [(50, 50), (100,)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__solver': ['adam' , 'lbfgs'] # lbfgs - often fails to converge even after increasing max_iter
        #'mlp__alpha': [0.0001, 0.001] # L2 regularization 
    }
    mlp_clf = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(max_iter=7000, random_state=42))])
    grid_search_mlp = run_grid_search(mlp_clf, mlp_params, X_train, y_train)

    # Best MLP model
    best_mlp = grid_search_mlp.best_estimator_
    y_val_pred_mlp = best_mlp.predict(X_val)
    mlp_accuracy = accuracy_score(y_val, y_val_pred_mlp)
    mlp_report = classification_report(y_val, y_val_pred_mlp, output_dict=True)

    print("Best MLP Parameters:", grid_search_mlp.best_params_)
    print("MLP Validation Accuracy:", mlp_accuracy)
    print(classification_report(y_val, y_val_pred_mlp))

    print(gridsearch2table(grid_search_mlp))