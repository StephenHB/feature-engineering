"""Classification methods for the project."""
import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def lgbm_classification(x, y):
    """Train and evaluate a LightGBM classifier."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = lgb.LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, y_pred, y_test


def train_and_evaluate(model, x, y):
    """Train a model and return accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Example usage:
# model, acc, preds, y_test = lgbm_classification(x, y)
# print(f"Test accuracy: {acc:.4f}") 