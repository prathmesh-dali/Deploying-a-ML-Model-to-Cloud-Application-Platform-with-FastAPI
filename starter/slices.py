import pandas as pd
import os
import pickle

from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics

def compute_slice_metric(model, encoder, binarizer, categorical_features, data):
    slice_metric = {}
    for feature in categorical_features:
        slice_metric[feature] = {}
        for val in data[feature].unique():
            X = data[data[feature] == val]
            X_slice, y_slice , *_ = process_data(X, categorical_features, label="salary", training=False, encoder=encoder, lb=binarizer)
            y_pred = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, y_pred)
            result = {"Precision" : precision, "Recall": recall, "Fbeta": fbeta}
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

    file = "./data/cleaned_data.csv"
    data = pd.read_csv(file)

    file_dir = os.path.dirname(__file__)


    model = pickle.load(open(os.path.join(file_dir,"model/model.pkl"), "rb"))
    encoder = pickle.load(open(os.path.join(file_dir,"model/encoder.pkl"), "rb"))
    binarizer = pickle.load(open(os.path.join(file_dir,"model/binarizer.pkl"), "rb"))

    compute_slice_metric(model, encoder, binarizer, cat_features, data)