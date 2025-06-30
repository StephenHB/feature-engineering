# Feature Engineering for Credit Score Datasets

This repository contains various methods for feature engineering applied to credit score datasets. The goal is to provide a modular and extensible framework for experimenting with different feature engineering techniques and validating their effectiveness on common classification models.

## Project Structure

- `data/`: Contains downloaded datasets from Kaggle and processed data files.
- `src/`: Source code for feature engineering methods, validation modules, and utilities.
- `requirements.txt`: List of required Python packages.

## Planned Methods

### Missing Value Imputation
- KNN Imputation
- missForest

### Variable Transformation
- Weight-of-Evidence (WoE)
- Box-Cox Transformation
- Yeo-Johnson Transformation
- Robust Scaling
- One-Hot Encoding

### Dimensionality Reduction & Feature Selection
- PCA
- SVD
- Linear Discriminant Analysis (LDA)
- t-SNE

### Feature Validation
- Compare engineered features against benchmarks (e.g., moving average, percentiles)
- Evaluate using classification models (LGBM, XGBoost)

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the dataset using KaggleHub:
   ```python
   import kagglehub
   kagglehub.dataset_download("parisrohan/credit-score-classification")
   ```
3. Explore and run feature engineering methods in the `src/` folder.
