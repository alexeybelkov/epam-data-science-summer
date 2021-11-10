import pandas as pd
import numpy as np
from  torch.nn import LSTM
import lightgbm
import xgboost
from sklearn.metrics import mean_squared_error as MSE

def MAPE(y, y_hat):
    return np.abs(y - y_hat)/y_hat

def RMSE(y, y_hat):
    return np.sqrt(MSE(y,y_hat))

