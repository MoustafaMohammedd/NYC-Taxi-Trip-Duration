import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from geopy import distance
from geopy.point import Point
import math

from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_squared_error
import xgboost as xgb

def predict_eval(model, data_preprocessed, target,name) -> str:
    y_train_pred = model.predict(data_preprocessed)
    mse = mean_squared_error(target, y_train_pred)
    rmse=mse**.5
    r2 = r2_score(target, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")
    return rmse, r2, f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}"

def log_transform(x):
    return np.log1p(np.maximum(x, 0))

def with_suffix(_, names: list[str]):  # https://github.com/scikit-learn/scikit-learn/issues/27695
    return [name + '__log' for name in names]

def pipeline_train(train, val):

    cat_features=['vendor_id', 'passenger_count','store_and_fwd_flag','pickup_hour', 'pickup_month', 'pickup_day', 'pickup_dayofweek', 'is_weekend', 'range_minutes','range_hours', 'range_months']
    num_features =[ 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude','distance_haversine', 'direction','distance_manhattan']

    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)

    train_features = num_features + cat_features

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=4)),
        ('log', LogFeatures)

    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features ),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough'
    )

    data_preprocessor = Pipeline(steps=[
        ('preprocessor', column_transformer)
    ])


    train_preprocessed = data_preprocessor.fit_transform(train[train_features])
    val_preprocessed = data_preprocessor.transform(val[train_features])



    ridge = Ridge(alpha=1, random_state=42)
    ridge.fit(train_preprocessed, train['log_trip_duration'])


    xgbr=xgb.XGBRegressor()
    xgbr.fit(train_preprocessed, train['log_trip_duration'])

    train_rmse_ridge, train_r2_ridge, _ = predict_eval(ridge, train_preprocessed, train['log_trip_duration'], "train_ridge")
    val_rmse_ridge, val_r2_ridge, _ = predict_eval(ridge, val_preprocessed, val['log_trip_duration'], "val_ridge")

    train_rmse_xgbr, train_r2_xgbr, _ = predict_eval(xgbr, train_preprocessed, train['log_trip_duration'], "train_xgbr")
    val_rmse_xgbr, val_r2_xgbr, _ = predict_eval(xgbr, val_preprocessed, val['log_trip_duration'], "val_xgbr")

    return ridge,xgbr, train_features, data_preprocessor, train_rmse_ridge, train_r2_ridge, val_rmse_ridge, val_r2_ridge,train_rmse_xgbr, train_r2_xgbr,val_rmse_xgbr, val_r2_xgbr

df_train=prepare_data(df_train)
df_val=prepare_data(df_val)

ridge,xgbr, train_features, data_preprocessor, train_rmse_ridge, train_r2_ridge, val_rmse_ridge, val_r2_ridge,train_rmse_xgbr, train_r2_xgbr,val_rmse_xgbr, val_r2_xgbr = pipeline_train(df_train, df_val)


# train_ridge RMSE = 0.4432 - R2 = 0.6891
# val_ridge RMSE = 0.4478 - R2 = 0.6867
# train_xgbr RMSE = 0.3826 - R2 = 0.7683
# val_xgbr RMSE = 0.4012 - R2 = 0.7485

df_test=prepare_data(df_test)

test_preprocessed=data_preprocessor.transform(df_test[train_features])

test_rmse_ridge, test_r2_ridge, _ = predict_eval(ridge, test_preprocessed, df_test['log_trip_duration'], "test_ridge")
test_rmse_xgbr, test_r2_xgbr, _ = predict_eval(xgbr, test_preprocessed, df_test['log_trip_duration'], "test_xgbr")

# test_ridge RMSE = 0.4426 - R2 = 0.6908
# test_xgbr RMSE = 0.3952 - R2 = 0.7534

