"""
Generates model stats on slices of data for categorical features
author: Prathmesh Dali
Date: March 2024
"""

import os
import pickle

import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics


def compute_slice_metric(
        model,
        encoder,
        binarizer,
        categorical_features,
        data):
    """Saves and returns model metrics for slices of categorical features"""
    slice_metric = {}
    for feature in categorical_features:
        slice_metric[feature] = {}
        for val in data[feature].unique():
            X = data[data[feature] == val]
            X_slice, y_slice, *_ = process_data(
                X, categorical_features, label="salary",
                training=False, encoder=encoder, lb=binarizer)
            y_pred = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, y_pred)
            result = {"Precision": precision, "Recall": recall, "Fbeta": fbeta}
            slice_metric[feature][val] = result
            print(f"Metric for {feature} = {val} = {result}")

    with open('slice_output.txt', 'w') as f:
        for feature, value in slice_metric.items():
            for key, value in value.items():
                f.write(f"{feature} = {key}: {value}")
                f.write("\n")
    return slice_metric


if __name__ == "__main__":

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    file_dir = os.path.dirname(__file__)
    file = os.path.join(file_dir, "./data/cleaned_data.csv")
    data = pd.read_csv(file)

    file_dir = os.path.dirname(__file__)

    with open(os.path.join(file_dir, "model/model.pkl"), "rb") as model_file:
        model = pickle.load(model_file)
    with open(os.path.join(file_dir, "model/encoder.pkl"), "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    with open(os.path.join(file_dir, "model/binarizer.pkl"), "rb") as lb_file:
        binarizer = pickle.load(lb_file)

    compute_slice_metric(model, encoder, binarizer, cat_features, data)
