"""Load credit data utilities."""
import os
import kagglehub
import pandas as pd

def download_credit_data():
    """
    Downloads the credit score classification dataset from Kaggle using kagglehub.
    Returns the path to the downloaded dataset directory.
    """
    dataset_path = kagglehub.dataset_download("parisrohan/credit-score-classification")
    return dataset_path

def load_credit_data(filename="train.csv"):
    """
    Loads the specified CSV file from the downloaded dataset into a pandas DataFrame.
    By default, loads 'train.csv'.
    """
    dataset_dir = download_credit_data()
    file_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist. Please check the filename.")
    return pd.read_csv(file_path)

if __name__ == "__main__":
    df = load_credit_data()
    print(df.head()) 