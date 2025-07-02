import pandas as pd
from typing import Dict


def detect_schema(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect the schema of a pandas DataFrame, mapping each column to a feature type.
    Types include: 'numeric', 'categorical', 'datetime', 'boolean', 'unknown'.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        Dict[str, str]: Mapping from column names to detected types.
    """
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
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
            # Check if all non-null values are strings
            non_null = df[col].dropna()
            all_str = bool(
                non_null.apply(lambda x: isinstance(x, str)).all()
            ) if not non_null.empty else False
            if all_str:
                schema[col] = 'str'
            else:
                schema[col] = 'categorical'
        else:
            schema[col] = 'unknown'
    return schema


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
        # ID columns: all unique, not datetime
        if nunique == n_rows and not pd.api.types.is_datetime64_any_dtype(dtype):
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
        # Categorical columns: object/category, not id, not binary
        elif (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)) and 2 < nunique < n_rows:
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
