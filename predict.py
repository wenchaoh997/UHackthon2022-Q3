import warnings
warnings.filterwarnings("ignore")
import os
os.system("pip install lightgbm")
os.system("pip install xgboost")
os.system("pip install catboost")
os.system("pip3 install autogluon")

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from joblib import dump, load
import pickle
import random
import sys

import time
import gc

import catboost as ctb
import xgboost
from xgboost import XGBRegressor
import lightgbm as lgb
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN = True
vectorizer = load("utils/TfidfVectorizer.joblib")
# vectorizer = load("utils/TfidfVectorizer")
# vectorizer = pickle.load(open("utils/vectorizer.pickle", "rb"))

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
            # assert new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "launch_date"].values == df.loc[i, "launch_date"]
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
#             new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "material_name_zh"] = list(
#                 set(new_df.loc[new_df["uuid"]==df.loc[i, "uuid"], "material_name_zh"] + " " + df.loc[i, "material_name_zh"])
#             )
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

# combine all chinese thing
def comb_zh(df):
    """
    material_name_zh
    ingredient
    """
    df["m&ingr"] = "0"
    n = len(df)
    for i in range(n):
        df.loc[i, "m&ingr"] = df.loc[i, "material_name_zh"] + "锛�" + df.loc[i, "ingredient"]
    return df

def feature_engineering(info_df, sales_df, save=False) -> pd.DataFrame:
    # channel
    sales_df.loc[sales_df["channel"]=="EC", "channel"] = 0
    sales_df.loc[sales_df["channel"]=="DT", "channel"] = 1
#     sales_df.channel = sales_df.channel.astype(int)
    # category
    info_df.drop(["category"], axis=1, inplace=True)
    # brand
    info_df[["DOVE", "LUX", "VASELINE"]] = 0
    info_df.DOVE = info_df.DOVE.astype(int)
    info_df.LUX = info_df.LUX.astype(int)
    info_df.VASELINE = info_df.VASELINE.astype(int)
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
    # comb name and ingr
#     info_df = comb_zh(info_df)
    
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


## ------------------- predict functions ------------------------

def predict(mm: str, model_path: str, data):
    if mm == "lightgbm":
        model = lgb.Booster(model_file=model_path)
    elif mm == "xgboost":
        model = xgboost.Booster(model_file=model_path)
    elif mm == "catboost":
        model = ctb.CatBoostRegressor()
        model.load_model(model_path)
    elif mm == "autogluon":
        model = TabularPredictor.load(model_path)
        
    try:
        data.drop("sales_value", axis=1, inplace=True)
    except:
        pass
    
    if mm == "lightgbm":
        pred = model.predict(data, num_iteration=model.best_iteration)
    else:
        # best_iteration = model.get_booster().best_ntree_limit
        # pred = model.predict(data, ntree_limit=best_iteration)
        pred = model.predict(data)
        
    pred = minmax_inv(pred, _min, _max)
    return pred
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: predict.py <product_info> <sales>')
    
    tic = time.perf_counter()
    
    # test df
    predict_product_info = pd.read_csv(sys.argv[1])
    predict_sales = pd.read_csv(sys.argv[2])
    
    # cols = ['channel', 'sales_period_', 'sales_value', 
    #       'DOVE', 'LUX', 'VASELINE', 'mentioned', 'yy', 'mm', 'dd', 
    #       'dToM1', 'dToM2', 'dToM10', 'dToM12'] + [str(x) for x in range(100)]
    
    cols = ['channel', 'sales_period_', 'sales_value', 
          'DOVE', 'LUX', 'VASELINE', 'mentioned', 'yy', 'mm', 'dd', 
          'dToM1', 'dToM2', 'dToM10', 'dToM12']
    
    new_info, new_sales = feature_engineering(predict_product_info, predict_sales, save=False)
    X_test = pd.merge(new_sales, new_info, on="uuid", how="left")
    # fillin null
    X_test.fillna(method="ffill", inplace=True)

    # tmp_w2v = vectorizer.transform(X_test["ingredient"])
    # tmp_w2v = pd.DataFrame(tmp_w2v.todense(), columns=[str(x) for x in range(100)])
    # X_test = pd.merge(X_test, tmp_w2v, left_index=True, right_index=True)
    
    # prepare ag test set
    scols = ["mentioned", "dToM1", "dToM2", "dToM10", "dToM12", "mm", "dd"]
    scaler = load('./scalers/lgbm.joblib')
    out = scaler.transform(X_test[scols])
    X_test[scols] = out
    try:
        os.mkdir("__tmp")
    except:
        pass
    X_test.to_csv("__tmp/test.csv", index=False)
    
    ##Please place your inference code here.
    
    pred = []
    pred_lgb = []
    pred_ens = []
    pred_ag = []
    
    # ag
    ag_test = TabularDataset("__tmp/test.csv")
    # tmp_ag = predict("autogluon", "./ag/", ag_test[cols])
    # pred_ag.append(tmp_ag)
    tmp_ag = predict("autogluon", "./agsn/", ag_test[cols])
    pred_ag.append(tmp_ag)
    pred_ag = np.mean(np.column_stack(pred_ag), axis=1)
    pred.append(pred_ag)
    ## ==================================================
    cols = ['sales_value',
         'mentioned',
         'yy',
         'mm',
         'dd',
         'dToM1',
         'dToM2',
         'dToM10',
         'dToM12']
    # lgb
    tmp_lgb = predict("lightgbm", "modelsn-1.76.txt", X_test[cols])
    pred_lgb.append(tmp_lgb)
    tmp_lgb = predict("lightgbm", "modelsn-1.85.txt", X_test[cols])
    pred_lgb.append(tmp_lgb)
    # tmp_lgb = predict("lightgbm", "modelv1-2.040.txt", X_test[cols])
    # pred_lgb.append(tmp_lgb)
    # tmp_lgb = predict("lightgbm", "modelv1-2.07.txt", X_test[cols])
    # pred_lgb.append(tmp_lgb)
    pred_lgb = np.mean(np.column_stack(pred_lgb), axis=1)
    pred.append(pred_lgb)
    
    # prediction
    pred = np.mean(np.column_stack(pred), axis=1)
    
    predict_sales = pd.read_csv(sys.argv[2])
    predict_sales["sales_value"] = pred
    
    predict_sales.to_csv('result.csv')
    
    try:
        print("Score: ", mean_squared_error(pred, X_test["sales_value"]))
    except:
        pass
    toc = time.perf_counter()
    print(f"Inference completed in {toc - tic:0.4f} seconds")
