# Feature Engineering for House Price Prediction Model
## Overview
I made this project with the purpose of learning a few feature engineering methodologies and workflows. In this project, I strive to minimize the Root Mean Squared Logarithmic Error using more than one popular Feature Engineering method
## Skills Implemented
1. Using Mutual Importance Scores ```mutual_info_regression``` to make decisions regarding which features to implement
2. Implementation of a powerful machine learning model ```XGBRegressor```
3. Scaling of Important Numerical Features
4. K-Means Clustering to add an integer-based ```Cluster Number``` column
5. Principal Component Analysis for Dimensionality Reduction
6. Target Encoding on a few High Cardinality columns
## Results
```
RMSLE Before Feature Engineering:  0.14336777904947118
Correlation with SalePrice:

GarageArea      0.640138
YearRemodAdd    0.532974
TotalBsmtSF     0.632529
GrLivArea       0.706780
dtype: float64
Feature1    0.873029
Feature3    0.341337
Feature2    0.262388
Feature4    0.087857
Name: MI Scores, dtype: float64
RMSLE After Feature Engineering:  0.14116773093720625
```
