"""Imputation methods for feature engineering."""
import pandas as pd
from sklearn.impute import KNNImputer

def knn_impute(df, n_neighbors=5, weights='uniform', columns=None):
    """
    Impute missing values in a DataFrame using KNN imputation.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values.
        n_neighbors (int): Number of neighboring samples to use for imputation.
        weights (str): Weight function used in prediction. Possible values: 'uniform', 'distance'.
        columns (list or None): List of columns to impute. If None, all columns are imputed.
    
    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    if columns is not None:
        df_copy = df.copy()
        df_copy[columns] = imputer.fit_transform(df_copy[columns])
        return df_copy
    else:
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

# Example usage:
# df = pd.read_csv('your_data.csv')
# df_imputed = knn_impute(df, n_neighbors=3) 

def impute_data(df, method):
    """Impute missing values in a DataFrame based on the specified method.

    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values.
        method (str): Imputation method. Possible values: 'mean', 'median', 'knn'.

    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    if method == 'mean':
        return df.fillna(df.mean())
    if method == 'median':
        return df.fillna(df.median())
    if method == 'knn':
        imputer = KNNImputer()
        return pd.DataFrame(
            imputer.fit_transform(df), columns=df.columns
        )
    return df 