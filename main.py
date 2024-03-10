"""
Code to expose api for model
author: Prathmesh Dali
Date: March 2024
"""

import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference


class InputData(BaseModel):
    """
    This class creates type hint for api
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "age": 47,
                    "workclass": "Self-emp-not-inc",
                    "fnlgt": 149116,
                    "education": "Masters",
                    "education_num": 14,
                    "marital_status": "Never-married",
                    "occupation": "Prof-specialty",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Female",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 50,
                    "native_country": "United-States"
                    },

            {
                'age': 30,
                'workclass': "State-gov",
                'fnlgt': 141297,
                'education': "Bachelors",
                'education_num': 13,
                'marital_status': "Married-civ-spouse",
                'occupation': "Prof-specialty",
                'relationship': "Husband",
                'race': "Asian-Pac-Islander",
                'sex': "Male",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 40,
                'native_country': "India"
            }
        ]
    }}


app = FastAPI()


@app.get("/")
async def root():
    """
    This api on get call at root returns welcome string
    """
    return "Welcome to the app."

file_dir = os.path.dirname(__file__)


with open(os.path.join(file_dir, "model/model.pkl"), "rb") as model_file:
    model = pickle.load(model_file)
with open(os.path.join(file_dir, "model/encoder.pkl"), "rb") as encoder_file:
    encoder = pickle.load(encoder_file)
with open(os.path.join(file_dir, "model/binarizer.pkl"), "rb") as lb_file:
    binarizer = pickle.load(lb_file)

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


@app.post("/predict")
async def predict(data: InputData):
    """
    This api returns the prediction of model on given input through post call.
    """
    data = pd.DataFrame(data.__dict__, [0])
    data.columns = [c.replace("_", "-") for c in data.columns]
    data, *_ = process_data(data, categorical_features=cat_features,
                            training=False, encoder=encoder, lb=binarizer)
    pred = inference(model, data)
    return binarizer.inverse_transform(pred)[0]
