"""
Week 4 - Predictive Modeling and Optimization in Logistics
Synthetic dataset: predict delivery time and use predicted risk for capacity allocation.
"""
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import linprog

DATA=Path("data/hypothetical_logistics_prediction_data.csv")
df=pd.read_csv(DATA)

X=df.drop(columns="delivery_days")
y=df["delivery_days"]

categorical=["region","transport_mode","priority","weather"]
numeric=["shipment_volume","distance_km","congestion_index"]

preprocess=ColumnTransformer([
    ("cat",OneHotEncoder(handle_unknown="ignore"),categorical)
],remainder="passthrough")

model=Pipeline([
    ("preprocess",preprocess),
    ("model",RandomForestRegressor(
        n_estimators=250,max_depth=10,random_state=42,n_jobs=-1))
])

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=.20,random_state=42
)

model.fit(X_train,y_train)
pred=model.predict(X_test)

mae=mean_absolute_error(y_test,pred)
rmse=mean_squared_error(y_test,pred)**.5
r2=r2_score(y_test,pred)

print("MAE:",round(mae,3))
print("RMSE:",round(rmse,3))
print("R2:",round(r2,3))

# 5-fold cross-validation
cv=KFold(n_splits=5,shuffle=True,random_state=42)
grid=GridSearchCV(
    model,
    {"model__max_depth":[6,10,14],
     "model__min_samples_leaf":[1,3,5]},
    cv=cv,scoring="neg_root_mean_squared_error"
)
grid.fit(X_train,y_train)

best_model=grid.best_estimator_
print("Best parameters:",grid.best_params_)

# Illustrative linear optimization:
# minimize risk-adjusted regional capacity cost
# subject to total capacity = 1000 and 120 <= each region <= 350.
cost=[10.5,9.5,11.2,9.8]
result=linprog(
    cost,
    A_eq=[np.ones(4)],
    b_eq=[1000],
    bounds=[(120,350)]*4,
    method="highs"
)
print("Optimized allocation:",result.x)
