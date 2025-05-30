
# Customer Transaction Prediction (PRCP-1003)

## Problem Statement
Predict whether a customer will make a transaction in the future using anonymized banking data.

## Dataset
- Features: 200 anonymized numerical attributes
- Target: Binary value (1 = will transact, 0 = will not transact)

## Steps Performed
1. Loaded and validated data.
2. Handled class imbalance using stratified train/test split.
3. Trained and evaluated multiple models:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LightGBM
4. Evaluated using Accuracy and ROC-AUC.

## Best Model
- Selected based on highest ROC-AUC score.

## Challenges
- Lack of feature names made EDA irrelevant.
- Class imbalance handled using ROC-AUC.
- Chose models robust to feature scale and sparsity.

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, lightgbm
