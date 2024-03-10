# Script to train machine learning model.

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle
import os

file_dir = os.path.dirname(__file__)
file = os.path.join(file_dir, "../data/cleaned_data.csv")
# Add code to load in the data.
data = pd.read_csv(file)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, *_ = process_data(test, categorical_features=cat_features, label="salary", training= False, encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train, y_train)

y_pred = inference(model, X_test)

precision, recall, fbeta =  compute_model_metrics(y_test, y_pred)

result = {"Precision" : precision, "Recall": recall, "Fbeta": fbeta}

print(result)
print(model.best_params_)

pickle.dump(model, open(os.path.join(file_dir, "../model/model.pkl"),"wb"))
pickle.dump(encoder, open(os.path.join(file_dir, "../model/encoder.pkl"),"wb"))
pickle.dump(lb, open(os.path.join(file_dir, "../model/binarizer.pkl"),"wb"))

with open(os.path.join(file_dir, '../result.txt'), 'w') as f:
    f.write(f"{model.best_params_}")
    f.write("\n")
    f.write(f"{result}")
    f.write("\n")