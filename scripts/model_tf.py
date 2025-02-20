#!/usr/bin/env python3

import os
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path = os.getcwd()
parent = os.path.dirname(path) # extract parent dir in cross platform way

path_to_plots = os.path.join(parent, "plots")
path_to_data = os.path.join(parent, "data")
path_to_model = os.path.join(parent, "model")


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(path_to_data, "train_cleaned.csv"))
    test = pd.read_csv(os.path.join(path_to_data, "test_cleaned.csv"))

    # Separate features and target variable
    X = train.drop("Survived", axis=1)
    y = train["Survived"]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        # Dropout(0.3),
        Dense(64, activation='relu'),
        # Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Save the model structure and weights
    model_json = model.to_json()
    with open(os.path.join(path_to_model, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # model.save_weights(os.path.join(path_to_model, "model.h5"))
    model.save(os.path.join(path_to_model, "model.h5"))

    # Calculate scores
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Save the model summary to a text file
    # doesn't work on new tf
    # with open(os.path.join(path_to_model, "model_summary.txt"), "w") as summary_file:
    #     model.summary(print_fn=lambda x: summary_file.write(x + "\n"))
    print(model.summary())

