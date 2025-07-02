"""
Module for automatic feature grouping utilities.
"""
import pandas as pd
from typing import Dict, Any
import numpy as np


def get_type_counts(series):
    """Return value counts of types in a pandas Series (excluding nulls)."""
    return series.dropna().apply(lambda x: type(x)).value_counts()

def get_majority_type(type_counts):
    """Return the most common type from a type_counts Series, or None if not found."""
    if not type_counts.empty and isinstance(type_counts.idxmax(), type):
        return type_counts.idxmax()
    return None

def handle_object_column(series):
    """
    Try to convert an object column to a more specific type.
    Returns the converted Series and the detected type as a string.
    """
    # Try numeric
    try:
        converted = pd.to_numeric(series, errors='coerce')
        if pd.api.types.is_numeric_dtype(converted):
            return converted, 'numeric'
    except Exception:
        pass

    # Try datetime
    try:
        converted = pd.to_datetime(series, errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(converted):
            return converted, 'datetime'
    except Exception:
        pass

    # Try boolean
    try:
        converted = series.map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False, '1': True, '0': False})
        if converted.dropna().isin([True, False]).all():
            return converted, 'boolean'
    except Exception:
        pass

    # Try category for low cardinality
    nunique = series.nunique(dropna=True)
    if nunique < 0.5 * len(series):
        try:
            converted = series.astype('category')
            return converted, 'categorical'
        except Exception:
            pass

    # Try cleaning and converting to numeric (more aggressive)
    try:
        cleaned = series.astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        # Convert empty strings to NaN
        cleaned = cleaned.replace('', np.nan)
        numeric = pd.to_numeric(cleaned, errors='coerce')
        # Only proceed if numeric is a pandas Series
        if isinstance(numeric, pd.Series) and numeric.notnull().sum() > 0.5 * len(series):
            return numeric, 'numeric'
    except Exception:
        pass

    # Fallback: convert to string
    converted = series.astype(str)
    return converted, 'str'

def detect_and_adjust_data_schema(df: pd.DataFrame):
    """
    Detect and adjust the schema of a pandas DataFrame. For object columns, attempt to convert to the majority non-null type in that column.
    Returns the adjusted DataFrame and a schema dict mapping column names to detected types.
    """
    schema = {}
    df_adj = df.copy()
    for col in df_adj.columns:
        dtype = df_adj[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            schema[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema[col] = 'datetime'
        elif pd.api.types.is_bool_dtype(dtype):
            schema[col] = 'boolean'
        elif pd.api.types.is_string_dtype(dtype):
            schema[col] = 'str'
        elif pd.api.types.is_categorical_dtype(dtype):
            schema[col] = 'categorical'
        elif pd.api.types.is_object_dtype(dtype):
            df_adj[col], schema[col] = handle_object_column(df_adj[col])
        else:
            schema[col] = 'unknown'
    return df_adj, schema


def group_columns(df: pd.DataFrame) -> dict:
    """
    Pre-scan a pandas DataFrame and classify columns into:
    id_cols, date_cols, continuous_cols, binary_cols, categorical_cols, other_cols.
    
    Returns:
        dict: {id_cols, date_cols, continuous_cols, binary_cols, categorical_cols, other_cols}
    """
    id_cols = []
    date_cols = []
    continuous_cols = []
    binary_cols = []
    categorical_cols = []
    other_cols = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        dtype = series.dtype
        col_lower = col.lower()
        # ID columns: name includes 'name' or 'id', or nearly all unique, not numeric, not datetime
        if (
            ("name" in col_lower or "id" in col_lower) and not pd.api.types.is_numeric_dtype(dtype)
        ) or (
            (nunique == n_rows or nunique > 0.95 * n_rows)
            and not pd.api.types.is_datetime64_any_dtype(dtype)
            and not pd.api.types.is_numeric_dtype(dtype)
        ):
            id_cols.append(col)
        # Date columns
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            date_cols.append(col)
        # Binary columns: exactly 2 unique values
        elif nunique == 2:
            binary_cols.append(col)
        # Continuous columns: numeric, more than 2 unique values
        elif pd.api.types.is_numeric_dtype(dtype) and nunique > 2:
            continuous_cols.append(col)
        # Categorical columns: object/category, not id, not binary, not nearly unique
        elif (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)) and 2 < nunique < 0.95 * n_rows:
            categorical_cols.append(col)
        else:
            other_cols.append(col)
    return {
        'id_cols': id_cols,
        'date_cols': date_cols,
        'continuous_cols': continuous_cols,
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols,
        'other_cols': other_cols
    }

def manually_adjust_input_cols(grouped_cols: Dict[str, list], change: Dict[str, Any]) -> Dict[str, list]:
    """
    Move a column from one group to another in the grouped_cols dictionary based on the change instruction.
    Args:
        grouped_cols (dict): Dictionary of column groups, e.g., {'id_cols': [...], 'continuous_cols': [...], ...}
        change (dict): Instruction dict, e.g.,
            {
                'current_col': 'id_cols',
                'new_col': 'continuous_cols',
                'val': 'Monthly_Balance'
            }
    Returns:
        dict: Updated grouped_cols dictionary.
    """
    current_col = change.get('current_col')
    new_col = change.get('new_col')
    val = change.get('val')

    if current_col not in grouped_cols or new_col not in grouped_cols:
        print(f"[manual adjust] One of the groups '{current_col}' or '{new_col}' does not exist.")
        return grouped_cols

    if val in grouped_cols[current_col]:
        grouped_cols[current_col].remove(val)
        if val not in grouped_cols[new_col]:
            grouped_cols[new_col].append(val)
        print(f"Moved '{val}' from '{current_col}' to '{new_col}'.")
    else:
        print(f"[manual adjust] '{val}' not found in '{current_col}'. No change made.")
    return grouped_cols
