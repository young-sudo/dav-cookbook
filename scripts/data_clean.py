#!/usr/bin/env python3

import pandas as pd
import os

path = os.getcwd()
parent = os.path.dirname(path) # extract parent dir in cross platform way

path_to_plots = os.path.join(parent, "plots")
path_to_data = os.path.join(parent, "data")


if __name__ == "__main__":

    train = pd.read_csv(os.path.join(path_to_data, "train.csv"))
    test = pd.read_csv(os.path.join(path_to_data, "test.csv"))

    # Feature selection
    # - remove cabin, almost all cabin entries are nan
    # - remove ticket and name, they are non-informative for training
    # - remove rows with nan
    train_clean = train.drop(["PassengerId", "Cabin", "Ticket", "Name"], axis=1).dropna()
    test_clean = test.drop(["PassengerId", "Cabin", "Ticket", "Name"], axis=1).dropna()

    # - change sex to numerical, and call it gender
    gender_dict = {
        'male' : 0,
        'female' : 1
    }
    gender_df = pd.DataFrame.from_dict(gender_dict, orient='index', columns=["Gender"]).reset_index().rename({"index" : "Sex"}, axis=1)
    train_clean = train_clean.merge(gender_df, how="left", on="Sex").drop("Sex", axis=1)
    test_clean = test_clean.merge(gender_df, how="left", on="Sex").drop("Sex", axis=1)

    # - one-hot encode port of embarkment
    train_oh = pd.get_dummies(train_clean["Embarked"], dtype=int)
    train_clean = train_clean.join(train_oh).drop("Embarked", axis=1)
    test_oh = pd.get_dummies(test_clean["Embarked"], dtype=int)
    test_clean = test_clean.join(test_oh).drop("Embarked", axis=1)

    train_clean.to_csv(os.path.join(path_to_data, "train_cleaned.csv"), index=False)
    test_clean.to_csv(os.path.join(path_to_data, "test_cleaned.csv"), index=False)  
