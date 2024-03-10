"""
Tests functionality of apis
author: Prathmesh Dali
Date: March 2024
"""

import json
from fastapi.testclient import TestClient

from main import app, InputData

client = TestClient(app)


def test_get():
    """
    Test get call of api
    """
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == "Welcome to the app."


def test_post_1():
    """
    Test post call of api
    """
    data = json.dumps(
        InputData.model_config["json_schema_extra"]["examples"][0])
    res = client.post("/predict", data=data)
    assert res.status_code == 200
    assert res.json() == "<=50K"


def test_post_2():
    """
    Test post call of api
    """
    data = json.dumps(
        InputData.model_config["json_schema_extra"]["examples"][1])
    res = client.post("/predict", data=data)
    assert res.status_code == 200
    assert res.json() == ">50K"
