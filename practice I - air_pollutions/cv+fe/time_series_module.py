import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose

def lagged(data, col, lag):
    return data[col].shift(lag)

def RWF(data, col, lag):
    return data[col].rolling(lag)

def EWF(data, col, lag):
    return data[col].expanding(lag)

def get_season(x):      # 1 : winter , 2 : spring, 3 : summer, 4 : autumn
    if x in [12,1,2]:
        return 1
    if x in [3,4,5]:
        return 2
    if x in [6,7,8]:
        return 3
    if x in [9,10,11]:
        return 4

def get_dt_frame(data, name, begin, end, fill = 0):
    data[[name + '_' + str(t) for t in range(begin, end)]] = pd.DataFrame(data = fill,
                                                                          index = list(range(len(data))),
                                                                          columns = [name + '_' + str(t)
                                                                                     for t in range(begin, end)])

def get_date_time_features(data,dt_col, hour = True, day = True, month = False, season = False, year = False):
    if hour:
        get_dt_frame(data, 'h', begin = 0, end = 24, fill = 0)
        temp = data[dt_col].dt.hour
        for i in range(len(data)):
            data.loc[i, 'h_' + str(temp[i])] = 1

    if day:
        get_dt_frame(data, 'd', begin = 1, end = 32, fill = 0)
        temp = data[dt_col].dt.day
        for i in range(len(data)):
            data.loc[i, 'd_' + str(temp[i])] = 1

    if month:
        get_dt_frame(data, 'm', begin = 1, end = 13, fill = 0)
        temp = data[dt_col].dt.month
        for i in range(len(data)):
            data.loc[i, 'm_' + str(temp[i])] = 1

    if season:
        get_dt_frame(data, 's', begin = 1, end = 5, fill = 0)
        temp = data[dt_col].dt.month
        for i in range(len(data)):
            ssn_num = get_season(temp[i])
            data.loc[i , 's_' + str(ssn_num)] = 1

    if year:
        get_dt_frame(data, 'y', begin = 2010, end = 2012, fill = 0)
        temp = data[dt_col].dt.year
        for i in range(len(data)):
            data.loc[i, 'y_' + str(temp[i])] = 1

def get_outliers(data, *perc):
    Q1, Q3 = np.percentile(data, perc)
    IQR = Q3 - Q1
    dw, uw = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return np.logical_or(data <= dw, data >= uw)


def get_anomalies(data, scale, lag):
    rolling_mean = data.rolling(lag).mean()
    mae = mean_absolute_error(data[lag:], rolling_mean[lag:])
    deviation = np.std(data[lag:] - rolling_mean[lag:])
    lower_bond = rolling_mean - (mae + scale * deviation)
    upper_bond = rolling_mean + (mae + scale * deviation)
    return np.logical_or(data > upper_bond, data < lower_bond)

def broadcast(*LISTS):
    LISTS = list(map(lambda x : x if type(x) == list else [x], LISTS))
    k = max(map(len, LISTS))
    LISTS = list(map(lambda x : x * k if len(x) == 1 else x, LISTS))
    return list(LISTS)

def feature_extractor(features_dict, data, **parameters):
    #features_dict = {}
    for key in parameters.keys():
        if key not in features_dict.keys():
            features_dict[key] = []
        if key == 'lagged': # col, lag
            cols, lags = parameters[key]
            cols, lags = broadcast(cols, lags)
            for c, l in zip(cols, lags):
                features_dict[key].append('lag_' + c + '_' + str(l))
                data['lag_' + c + '_' + str(l)] = lagged(data, c, l)

        elif key == 'RWF': # col, lag, func
            cols, lags, funcs = parameters[key]
            cols, lags, funcs = broadcast(cols, lags, funcs)
            for c,l,f in zip(cols, lags, funcs):
                features_dict[key].append('rol_' + c + '_lag_' + str(l) + '_' + f.__name__)
                data['rol_' + c + '_lag_' + str(l) + '_' + f.__name__] = RWF(data, c, l).apply(f)

        elif key == 'EWF': # col, lag, func
            cols, lags, funcs = parameters[key]
            cols, lags, funcs = broadcast(cols, lags, funcs)
            for c, l, f in zip(cols, lags, funcs):
                features_dict[key].append('exp_' + c + '_lag_' + str(l) + '_' + f.__name__)
                data['exp_' + c + '_lag_' + str(l) + '_' + f.__name__] = EWF(data, c, l).apply(f)

        elif key == 'STL': # date_time column, columns , all_ = False/True, components = False, model (additive or multiplicative)
            dt, columns, all_, components, model = parameters[key][0], parameters[key][1], parameters[key][2], parameters[key][3], parameters[key][4]
            if all_:
                columns = columns if type(columns) == list else [columns]
                for col in columns:
                    temp_series = data[col].copy()
                    temp_series.index = dt
                    features_dict[key].extend(['trend_' + col, 'seasonal_' + col, 'resid_' + col])
                    data['trend_' + col] = seasonal_decompose(temp_series, extrapolate_trend = 'freq', model = model).trend.values
                    data['seasonal_' + col] = seasonal_decompose(temp_series, extrapolate_trend = 'freq', model = model).seasonal.values
                    data['resid_' + col] = seasonal_decompose(temp_series, extrapolate_trend = 'freq', model = model).resid.values
            elif components:
                columns, components = broadcast(columns, components)
                for col, comp in zip(columns, components):
                    temp_series = data[col].copy()
                    temp_series.index = dt
                    if comp == 'trend':
                        features_dict[key].append(comp + '_' + col)
                        data[comp + '_' + col] = seasonal_decompose(temp_series, extrapolate_trend = 'freq', model = model).trend.values
                    elif comp == 'seasonal':
                        features_dict[key].append(comp + '_' + col)
                        data[comp + '_' + col] = seasonal_decompose(temp_series, extrapolate_trend = 'freq', model = model).seasonal.values
                    elif comp == 'resid':
                        features_dict[key].append(comp + '_' + col)
                        data[comp + '_' + col] = seasonal_decompose(temp_series, extrapolate_trend = 'freq', model = model).resid.values
        elif key == 'anomalies': # data, columns,  scale, lag
            columns, scale, lag = parameters[key]
        elif key == 'outliers': # data, scale, *perc
            pass


def to_numpy(data, keepdims = True):
    if keepdims:
        return np.reshape(data.to_numpy(), (len(data), -1))
    else:
        return data.to_numpy()

def get_cv_matrices(train, test, features, targets, keepdims = True):
    X_train = to_numpy(train[features], keepdims = keepdims)
    Y_train = to_numpy(train[targets], keepdims = keepdims)
    X_test = to_numpy(test[features],  keepdims = keepdims)
    Y_test = to_numpy(test[targets], keepdims = keepdims)
    return X_train, Y_train, X_test, Y_test

class CrossValid:

    def __init__(self,train_size, test_size, min_period = 0):
        self.train_size = train_size
        self.test_size = test_size
        self.min_period = min_period
        self.n_splits = None

    def split(self, data_len, step = 1, gap = 0):
        train_begin, train_end = self.min_period + 0, self.min_period + self.train_size
        test_begin, test_end = self.min_period + self.train_size + gap, self.min_period + self.train_size + gap + self.test_size
        self.n_splits = int((data_len - self.min_period - gap - self.train_size - self.test_size)/step) + 1
        while test_end <= data_len:
            yield list(range(train_begin, train_end)), list(range(test_begin, test_end))
            train_begin += step
            train_end += step
            test_begin += step
            test_end += step