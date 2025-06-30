import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def lgbm_classification(X, y, test_size=0.2, random_state=42, **lgbm_params):
    """
    Train a simple LightGBM classifier and return accuracy on the test set.
    Ensures input data is numeric and contains no missing values. Drops unsupported columns and prints their names.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.
        **lgbm_params: Additional parameters for LGBMClassifier.
    
    Returns:
        model: Trained LGBMClassifier model.
        float: Accuracy score on the test set.
        np.ndarray: Predictions on the test set.
        np.ndarray: True labels for the test set.
    """
    # Identify non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"[LGBM] Dropping non-numeric columns: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
    # Ensure X is numeric and has no missing values
    X = X.apply(pd.to_numeric, errors='coerce')
    mask = X.notnull().all(axis=1) & pd.notnull(y)
    X = X[mask]
    y = y[mask]
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, y_pred, y_test

# Example usage:
# model, acc, preds, y_test = lgbm_classification(X, y)
# print(f"Test accuracy: {acc:.4f}") 