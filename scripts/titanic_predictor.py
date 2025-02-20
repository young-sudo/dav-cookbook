#!/usr/bin/env python3

import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import os

# scaling warning
import warnings
warnings.filterwarnings("ignore")

# Command-line tool for prediction
def get_user_input():
    print("\nDo you have what it takes to survive the Titanic?\nInput your data to find out :)")
    age = float(input("Age: "))
    gender = input("Gender (male/female): ").strip().lower()
    pclass = int(input("Socio-economic class (1/2/3): "))
    siblings = int(input("Number of siblings/spouses aboard: "))
    parch = int(input("Number of parents/children aboard: "))
    fare = float(input("Fare: "))
    embarked = input("Port of Embarkation (C/Q/S): ").strip().upper()

    gender_dict = {'male': 0, 'female': 1}
    embarked_dict = {'C': [1, 0, 0], 'Q': [0, 1, 0], 'S': [0, 0, 1]}

    user_input = [age, gender_dict[gender], pclass, siblings, parch, fare]
    user_input.extend(embarked_dict[embarked])
    
    return np.array(user_input).reshape(1, -1)

def predict_survival(input_data):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return prediction[0][0]

if __name__ == "__main__":
    input_data = get_user_input()

    # Load model
    with open("../model/model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("../model/model.h5")

    # Load scaler
    scaler = StandardScaler()
    train_clean = pd.read_csv("../data/train_cleaned.csv")
    X = train_clean.drop("Survived", axis=1)
    scaler.fit(X)

    prediction = predict_survival(input_data)
    if prediction > 0.5:
        print(f"Most likely you would survive the Titanic crash (prediction: SURVIVED {prediction:.4f})")
    else:
        print(f"Most likely you would not survive the Titanic crash (prediction: DEAD {prediction:.4f})")
