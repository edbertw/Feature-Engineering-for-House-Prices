import numpy as np
import pandas as pd
import warnings
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def score_model(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def apply_pca(X):
    if True:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(X) #PRINCIPAL COMPONENT ANALYSIS
    components = [f"Feature{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=components)
    return X_pca

data = pd.read_csv("ames.csv")
X = data.copy()
y = X.SalePrice
X_temp = data.copy()
y_temp = X_temp.pop("SalePrice")
print("RMSLE Before Feature Engineering: ", score_model(X_temp,y_temp))
features = ["LotArea","TotalBsmtSF","FirstFlrSF","SecondFlrSF","GrLivArea"]
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
kmeans = KMeans(n_clusters = 10,n_init = 10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled) #K-MEANS CLUSTERING

features = ["GarageArea","YearRemodAdd","TotalBsmtSF","GrLivArea"]
print("Correlation with SalePrice:\n")
print(data[features].corrwith(data.SalePrice))
X_1 = X.loc[:, features]
X_pca = apply_pca(X_1)
print(make_mi_scores(X_pca,y))
X = X.join(X_pca)
X.select_dtypes(["object"]).nunique() #Choose columns with high cardinality for Target Encoding

X_encode = X.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")

X_pretrain = X.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")

encoder = MEstimateEncoder(cols = ["Neighborhood","Exterior2nd"], m = 5.0)
encoder.fit(X_encode,y_encode)
X_train = encoder.transform(X_pretrain, y_train)
print("RMSLE After Feature Engineering: ",score_model(X_train,y_train))
