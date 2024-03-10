"""
Sample request for post call
author: Prathmesh Dali
Date: March 2024
"""

import json
import requests


data = json.dumps({
    "age": 27,
    "workclass": "Private",
    "fnlgt": 428030,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Craft-repair",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"})

res = requests.post(
    "https://deploying-a-ml-model-to-cloud.onrender.com/predict",
    data=data,
    timeout=50)

print("status_code", res.status_code)
print("result", res.json())
