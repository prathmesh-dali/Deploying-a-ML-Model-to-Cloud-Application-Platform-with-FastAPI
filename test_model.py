"""
Test for model file
author: Prathmesh Dali
Date: March 2024
"""

import os
import logging
import pickle
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import inference, compute_model_metrics, train_model
from starter.ml.data import process_data


@pytest.fixture(scope="module")
def data():
    """
    Fixture - will return the data as argument
    """
    file_dir = os.path.dirname(__file__)
    datapath = os.path.join(file_dir, "./data/cleaned_data.csv")
    return pd.read_csv(datapath)


@pytest.fixture(scope="module")
def cat_features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features


@pytest.fixture(scope="module")
def train_test_dataset(data, cat_features):
    """
    Fixture - returns cleaned train and test dataset to be used for model testing
    """
    train, test = train_test_split(data,
                                   test_size=0.20,
                                   random_state=10,
                                   stratify=data['salary']
                                   )
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    X_test, y_test, *_ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    return X_train, y_train, X_test, y_test


def test_train_model(train_test_dataset):
    """
    Check train model
    """

    X_train, y_train, *_ = train_test_dataset
    try:
        model = train_model(X_train, y_train)

    except Exception as err:
        raise err

    assert isinstance(model, GridSearchCV)

    assert isinstance(model.best_estimator_, RandomForestClassifier)


def test_inference(train_test_dataset):
    """
    Check inference function
    """
    _, _, X_test, _ = train_test_dataset

    savepath = "./model/model.pkl"
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as model_file:
            model = pickle.load(model_file)

        try:
            y_pred = inference(model, X_test)
        except Exception as err:
            logging.error(
                "Inference cannot be performed on saved model")
            raise err
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(X_test)
        assert ((y_pred == 1) | (y_pred == 0)).all()
    else:
        pass


def test_compute_model_metrics(train_test_dataset):
    """
    Check calculation of performance metrics function
    """
    _, _, X_test, y_test = train_test_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as model_file:
            model = pickle.load(model_file)
        y_pred = inference(model, X_test)

        try:
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        except Exception as err:
            logging.error(
                "Performance metrics cannot be calculated on data")
            raise err
        assert isinstance(precision, np.float64)
        assert isinstance(recall, np.float64)
        assert isinstance(fbeta, np.float64)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= fbeta <= 1
    else:
        pass
