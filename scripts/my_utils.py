#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import GridSearchCV

def gridsearch2table(grid_search):
    """
    Given a GridSearchCV object, return a DataFrame with the mean test accuracy
    for each combination of hyperparameters tested.
    """
    # Extract results from the GridSearchCV object
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Select relevant columns (parameters and mean test score)
    param_columns = [col for col in results.columns if col.startswith('param_')]
    score_column = ['mean_test_score']
    accuracy_table = results[param_columns + score_column]
    
    # Rename columns for better readability
    accuracy_table.columns = [col.replace('param_', '').replace('_', ' ').title() for col in accuracy_table.columns]
    
    # Rename the mean test score column
    accuracy_table = accuracy_table.rename(columns={'Mean Test Score': 'Mean Test Accuracy'})

    # Top 10 by accuracy
    accuracy_table = accuracy_table.sort_values("Mean Test Accuracy", ascending=False).head(10)

    # # Use Styler to apply background gradient
    # styled_table = accuracy_table.copy().style.background_gradient(cmap='viridis_r', subset=['Mean Test Accuracy'])
    # return styled_table
    
    return accuracy_table


# Function to run GridSearchCV and return results
def run_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search
