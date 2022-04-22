import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from joblib import dump, load
import random
import sys
import os
import time
import gc
os.system("pip install lightgbm")

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

GEN = 0
N_SAMPLE = 2500
TRAIN = False
MODEL_PATH = "modelv1.txt"
# MODEL_PATH = "modelv1-3.182.txt" # aug_1
# MODEL_PATH = "modelv1-2.12.txt" # aug_2
# MODEL_PATH = "modelv1-2.040.txt" # aug_2
MODEL_PATH = "modelv1-2.07.txt" # aug_2

##custom functions
def bar_code_helper(x: int) -> int:
    ##All values are 690
    x = x // 1e10
    return x

# for duplicated features
def duplicated_helper(df: pd.DataFrame) -> pd.DataFrame:
    idRecords = {}
    count = 0
    n = len(df)
    new_df = pd.DataFrame(columns=df.columns)
    for i in range(n):
        if df.loc[i, "uuid"] in idRecords.keys():
            assert new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "launch_date"].values == df.loc[i, "launch_date"]
            idRecords[df.loc[i, "uuid"]] += 1
            # ingredient
            new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "ingredient"] = list(
                set(df.loc[i, "ingredient"] + new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "ingredient"])
            )
            # material_name
            new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "material_name"] = list(
                set(new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "material_name"] + " " + df.loc[i, "material_name"])
            )
            # material_name_zh
            new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "material_name_zh"] = list(
                set(new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "material_name_zh"] + " " + df.loc[i, "material_name_zh"])
            )
        else:
            new_df.loc[count,] = df.loc[i,].tolist()
            idRecords[df.loc[i, "uuid"]] = 1
            count += 1
    new_df["mentioned"] = 1
    n = len(new_df)
    for i in range(n):
        new_df.loc[i, "mentioned"] = idRecords[new_df.loc[i, "uuid"]]
    return new_df

def ingredient_helper(x: str) -> str:
    x = x[1:-1].replace("'", "")
    x = x.replace("[", "")
    return x.replace("]", "")

def launch_date_helper(x: str) -> int:
    yy, mm, dd = x.split("-")
    return [int(yy), int(mm), int(dd)]
def mm_distance_helper1(source: float, target: int=1) -> float:
    return min(abs(source - target), abs(source - 12 - target))
def mm_distance_helper2(source: float, target: int=2) -> float:
    return min(abs(source - target), abs(source - 12 - target))
def mm_distance_helper10(source: float, target: int=10) -> float:
    return min(abs(source - target), abs(source - 12 - target))
def mm_distance_helper12(source: float, target: int=12) -> float:
    return min(abs(source - target), abs(source - 12 - target))

def feature_engineering(info_df, sales_df, save=False) -> pd.DataFrame:
    # channel
    sales_df.loc[sales_df["channel"]=="EC", "channel"] = 0
    sales_df.loc[sales_df["channel"]=="DT", "channel"] = 1
    # category
    info_df.drop(["category"], axis=1, inplace=True)
    # brand
    info_df[["DOVE", "LUX", "VASELINE"]] = 0
    info_df.loc[info_df["brand"]=="DOVE", "DOVE"] = 1
    info_df.loc[info_df["brand"]=="LUX", "LUX"] = 1
    info_df.loc[info_df["brand"]=="VASELINE", "VASELINE"] = 1
    # bar_code
    info_df["bar_code"] = info_df["bar_code"].apply(bar_code_helper)
    info_df.drop(["bar_code"], axis=1, inplace=True)
    # duplicated features
    info_df = duplicated_helper(info_df)
    # launch_date
    info_df[["yy", "mm", "dd"]] = info_df["launch_date"].apply(launch_date_helper).tolist()
    # sales_period_
    sales_df.loc[sales_df["sales_period_"]==6, "channel"] = 0
    sales_df.loc[sales_df["sales_period_"]==12, "channel"] = 1
    # distance to important month
    # im = [1, 2, 10, 12]
    info_df["dToM1"] = info_df["mm"].apply(mm_distance_helper1)
    info_df["dToM2"] = info_df["mm"].apply(mm_distance_helper2)
    info_df["dToM10"] = info_df["mm"].apply(mm_distance_helper10)
    info_df["dToM12"] = info_df["mm"].apply(mm_distance_helper12)
    # ingredient
    info_df["ingredient"] = info_df["ingredient"].apply(ingredient_helper)
    
    if save:
        info_df.to_csv("info_clean.csv", index=False)
        sales_df.to_csv("sales_clean.csv", index=False)
    return info_df, sales_df

# for sales_value
_min, _max = 2, 13
def minmax(df, col, _min, _max) -> pd.DataFrame:
    df[col] = (df[col] - _min) / (_max - _min)
    return df
def minmax_inv(x: list, _min, _max) -> list:
    n = len(x)
    out = x.copy()
    for i in range(n):
        out[i] = x[i] * (_max - _min) + _min
    return out

## ------------------- main functions ------------------------

def train(n_sample=N_SAMPLE):
    if GEN:
        model = load('ctgan_dump')
        samples = model.sample(n_sample)
        samples.to_csv('ctgan_aug_3.csv', index = False)
    # load df
    train_info = "./unilever-sales-data-training-v6/train_info.csv"
    train_sales = "./unilever-sales-data-training-v6/train_sales.csv"
    sample_path = "./gen/ctgan_aug_2.csv"
    info = pd.read_csv(train_info)
    sales = pd.read_csv(train_sales)
    samples = pd.read_csv(sample_path)
    
    new_info, new_sales = feature_engineering(info, sales, save=False)
    data = pd.merge(new_sales, new_info, on="uuid", how="left")
    ##fillin null
    data.fillna(method="ffill", inplace=True)
    
    train_cols = ['channel', 'sales_period_', 'sales_value', 
                  'DOVE', 'LUX', 'VASELINE', 'mentioned', 'yy', 'mm', 'dd', 
                  'dToM1', 'dToM2', 'dToM10', 'dToM12']
    train = data[train_cols]
    train[["channel", "DOVE", "LUX", "VASELINE"]] = train[["channel", "DOVE", "LUX", "VASELINE"]].astype(int)
    samples = samples.astype(train.dtypes)
    samples = samples.sample(n_sample)
    samples = samples.reset_index(drop=True)
    train_cols = ['sales_value', 
                  'mentioned', 'yy', 'mm', 'dd', 
                  'dToM1', 'dToM2', 'dToM10', 'dToM12']
    # Standardscaler
    train_length = len(train)
    tmp = pd.concat([train, samples])
    tmp = tmp.reset_index(drop=True)
    
    tmp = minmax(tmp, "sales_value", _min, _max)
    scols = ["mentioned", "dToM1", "dToM2", "dToM10", "dToM12", "mm", "dd"]
    
    scaler = preprocessing.StandardScaler()
    out = scaler.fit_transform(tmp[scols])
    dump(scaler, './scalers/lgbm.joblib')
    
    tmp[scols] = out
    train = tmp.loc[:train_length, train_cols].reset_index(drop=True)
    samples = tmp.loc[train_length:, train_cols].reset_index(drop=True)
    ##todo: train and valid
    
    # model and evaluation
    X_train, y_train = samples.drop("sales_value", axis=1), samples.sales_value
    X_test, y_test = train.drop("sales_value", axis=1), train.sales_value

    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, label=y_test)

    params = {
        'boosting_type': "gbdt",
        'metric': 'rmse',
        'learning_rate': 0.08,
        'max_depth': 10,
        'num_leaves': 3,
        'num_boost_round': 150,
        'force_col_wise': True,
        'objective': 'tweedie',
        'verbose': 0
    }

    gbm = lgb.train(params, train_data, valid_sets=[validation_data])

    # eval
    y_pred = gbm.predict(X_test)
    print("Out shape : ",len(y_pred))
    
    y_test_inv = minmax_inv(y_test, _min, _max)
    y_pred = minmax_inv(y_pred, _min, _max)
    print(mean_squared_error(y_test_inv, y_pred))
    
    gbm.save_model(MODEL_PATH)
    
    return gbm, train_cols

def predict(model, data):
    try:
        data.drop("sales_value", axis=1, inplace=True)
    except:
        pass
    scols = ["mentioned", "dToM1", "dToM2", "dToM10", "dToM12", "mm", "dd"]
    scaler = load('./scalers/lgbm.joblib')
    out = scaler.transform(data[scols])
    data[scols] = out
    
    pred = model.predict(data, num_iteration=model.best_iteration)
    pred = minmax_inv(pred, _min, _max)
    return pred
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: predict.py <product_info> <sales>')
    
    tic = time.perf_counter()
    
    # test df
    predict_product_info = pd.read_csv(sys.argv[1])
    predict_sales = pd.read_csv(sys.argv[2])
    
    if TRAIN:
        train_info = "./unilever-sales-data-training-v6/train_info.csv"
        train_sales = "./unilever-sales-data-training-v6/train_sales.csv"
        info = pd.read_csv(train_info)
        sales = pd.read_csv(train_sales)
    
        # pre-processing
        predict_product_info = pd.concat([info, predict_product_info])
        predict_product_info = predict_product_info.reset_index(drop=True)
        
    cols = ['sales_value',
         'mentioned',
         'yy',
         'mm',
         'dd',
         'dToM1',
         'dToM2',
         'dToM10',
         'dToM12']
    
    new_info, new_sales = feature_engineering(predict_product_info, predict_sales, save=False)
    X_test = pd.merge(new_sales, new_info, on="uuid", how="left")
    # fillin null
    X_test.fillna(method="ffill", inplace=True)
    
    ##Please place your inference code here.
    if TRAIN:
        model, cols = train()
    else:
        model = lgb.Booster(model_file=MODEL_PATH)
    pred = predict(model, X_test[cols])
    
    predict_sales = pd.read_csv(sys.argv[2])
    predict_sales["sales_value"] = pred
    
    predict_sales.to_csv('result.csv')
    
    toc = time.perf_counter()
    print(f"Inference completed in {toc - tic:0.4f} seconds")

