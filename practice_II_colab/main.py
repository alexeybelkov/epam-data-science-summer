import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error as MSE
from statsmodels.tsa.seasonal import seasonal_decompose

def WAPE(y, y_hat, eps=10**(-3)):
    return np.sum(np.abs(y - y_hat)) / (np.sum(np.abs(y))+eps)

def MAPE(y, y_hat, eps = 10**(-3)):
    y_c = y.copy()
    y_c[y_c == 0] += eps
    return np.mean(np.abs((y - y_hat) / y_c))

def RMSE(y, y_hat):
    return np.sqrt(MSE(y,y_hat))

def MAE(y, y_hat):
    return np.mean(np.abs(y-y_hat))

def get_stl(x, col, model, type_, extr_trend='freq'):
    m = None
    if model == 'A':
        m = 'additive'
    elif model == 'M':
        m = 'multiplicative'
    temp = seasonal_decompose(x, extrapolate_trend=extr_trend, model=m)
    if type_ == 'S':
        return temp.seasonal
    elif type_ == 'T':
        return temp.trend
    elif type_ == 'R':
        return temp.resids

def feature_extractor(dataframe, gcols, inplace=False, **parameters):
    data = dataframe if inplace else dataframe.copy()
    group = data.groupby(gcols)
    new_features_list = []
    for key in parameters.keys():
        if key == 'lagged':
            params = parameters['lagged']
            for col, lag in zip(params['columns'], params['lags']):
                new_feature_name = f'{col}_[{lag}]'
                if new_feature_name not in data.columns:
                    data[new_feature_name] = group[col].shift(lag)
                    new_features_list.append(new_feature_name)

        elif key == 'window':
            params = parameters['window']
            for col, type_, lag, func in zip(params['columns'], params['types'], params['lags'], params['funcs']):
                new_feature_name = f'{col}_{type_}[{lag}]_{func.__name__}'
                if new_feature_name not in data.columns:
                    if type_ == 'R':
                        data[new_feature_name] = group[col].apply(lambda x : x.rolling(lag).apply(func))
                    elif type_ == 'E':
                        data[new_feature_name] = group[col].apply(lambda x : x.expanding(lag).apply(func)) ####
                    new_features_list.append(new_feature_name)

        elif key == 'STL':
            '''
            model : A for additive, M for multiplicative
            types : S for seasonal, T for trend, R for resids
            '''
            params = parameters['STL']
            for col, model, type_ in zip(params['columns'], params['model'], params['types']):
                new_feature_name = f'{col}_{model}{type_}'
                if new_feature_name not in data.columns:
                    data[new_feature_name] = group[col].apply(lambda x : get_stl(x, col, model, type_))
                    new_features_list.append(new_feature_name)
    return data, new_features_list
