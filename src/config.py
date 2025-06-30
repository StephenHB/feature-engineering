# Base configuration parameters for the project
# KaggleHub dataset path
KAGGLEHUB_DATASET_PATH = "parisrohan/credit-score-classification"

# Local paths to CSV files
LOCAL_TRAIN_CSV = "/Users/stephenzhang/Downloads/credit_score_classification/train.csv"
LOCAL_TEST_CSV = "/Users/stephenzhang/Downloads/credit_score_classification/test.csv"

# Target column name
TARGET_COL = "Credit_Mix"

# Default features to use
DEFAULT_FEATURES = ["Month", "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts"]

# ID columns
ID_COLS = ["Customer_ID", "ID", "Name", "SSN"]

# Categorical columns
CAT_COLS = ["Occupation", "Payment_Behaviour"] 