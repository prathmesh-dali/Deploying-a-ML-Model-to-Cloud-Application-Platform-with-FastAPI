import json
from fastapi.testclient import TestClient

from main import app, InputData

client = TestClient(app)

def test_get():
    res=client.get("/")
    assert res.status_code == 200
    assert res.json() == "Welcome to the app"

def test_post_1():
    data = json.dumps(InputData.model_config["json_schema_extra"]["examples"][0])
    res = client.post("/predict", data=data)
    assert res.status_code == 200
    assert res.json()=="<=50K"
    
def test_post_2():
    data = json.dumps(InputData.model_config["json_schema_extra"]["examples"][1])
    res = client.post("/predict", data=data)
    assert res.status_code == 200
    assert res.json()=="<=50K"