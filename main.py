# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference

class InputData(BaseModel):
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

    model_config ={ "json_schema_extra" :{
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                    },
                {
                    'age':28,
                    'workclass':"Private", 
                    'fnlgt':338409,
                    'education':"Bachelors",
                    'education_num':13,
                    'marital_status':"Married-civ-spouse",
                    'occupation':"Prof-specialty",
                    'relationship':"Wife",
                    'race':"Black",
                    'sex':"Female",
                    'capital_gain':0,
                    'capital_loss':0,
                    'hours_per_week':40,
                    'native_country':"Cuba"
                    }
                ]
            }}

app = FastAPI()

@app.get("/")
async def root():
    return "Welcome to the app"

file_dir = os.path.dirname(__file__)


model = pickle.load(open(os.path.join(file_dir,"model/model.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(file_dir,"model/encoder.pkl"), "rb"))
binarizer = pickle.load(open(os.path.join(file_dir,"model/binarizer.pkl"), "rb"))

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
    data = pd.DataFrame(data.__dict__,[0])
    data.columns = [c.replace("_", "-") for c in data.columns]
    data, *_  = process_data(data, categorical_features=cat_features, training=False, encoder=encoder, lb=binarizer)
    pred = inference(model, data)
    return binarizer.inverse_transform(pred)[0]