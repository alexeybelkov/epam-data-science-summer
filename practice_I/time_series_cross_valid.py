from time_series_module import ForecastModel
import numpy as np
import matplotlib.pyplot as plt
import time_series_module as tsm
from functools import reduce
from pandas import Series as pandas_Series
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

def cross_valid(forecast_model, cv_dict, model, metrics):
    data, features, target = forecast_model.data, forecast_model.features, forecast_model.targets
    cv = cv_dict['cv']
    cv_split = list(cv.split(len(data), cv_dict['step'], cv_dict['gap']))
    
    results = {'mean_Y' : np.zeros((cv.train_size,1)),
               'mean_Y_pred' : np.zeros((cv.test_size, 1)), 
               'train_loss' : 0, 'test_loss' : 0, 'train_horizons' : [], 'test_horizons' : [],
               'mean_Y_train_pred' : np.zeros((cv.train_size, 1)), 'mean_Y_test' :np.zeros((cv.test_size, 1)), 'importances' : None}
    for train, test in cv_split:
        X_train, Y_train, X_test, Y_test = tsm.get_cv_matrices(data.loc[train], data.loc[test], features, target)
        fitted_model = model.fit(X_train, Y_train)
        #print(list(data.columns))
        Y_train_pred = np.reshape(fitted_model.predict(X_train), (len(X_train), 1))
        Y_pred = np.reshape(fitted_model.predict(X_test), (len(X_test), 1))
        results['train_loss'] += metrics(Y_train, Y_train_pred)
        results['test_loss'] += metrics(Y_test, Y_pred)
        results['mean_Y'] += Y_train
        results['mean_Y_train_pred'] += Y_train_pred
        results['mean_Y_test'] += Y_test
        results['mean_Y_pred'] += Y_pred
        results['importances'] = model.importances if results['importances'] is None else model.importances + results['importances']
    results['train_loss'] /= cv.n_splits
    results['test_loss'] /= cv.n_splits
    #results['mean_Y'] #/= cv.n_splits
    #results['mean_Y_pred'] #/= cv.n_splits
    results['cv_dict'] = cv_dict
    results['importances'] /= cv.n_splits
    return results

def run_cv(data, targets, horizons, CrossValid_params, ForecastModel_params, model, metrics):
    quality = {}
    features, date_time, prior_lag, post_lag = [ForecastModel_params['features'], ForecastModel_params['date_time'],
                                                ForecastModel_params['prior_lag'], ForecastModel_params['post_lag']]
    for tar in targets:
        quality[tar] = {'train_loss' : [], 'test_loss' : [], 'mean_Y' : [], 'mean_Y_pred' : [],
                        'mean_Y_train_pred' : [], 'mean_Y_test' : [], 'forecast_model' : [], 'importances' : []}
    for hor in range(1, horizons + 1):
        for tar in targets:
            forecast_model = ForecastModel(date_time, data, features + targets, [tar], prior_lag = list(range(hor,prior_lag + 1)),
                                           post_lag = post_lag)
            forecast_model.forecast_prep(ForecastModel_params['new_index'])
            train_size, test_size, min_period, step = [CrossValid_params['train_size'], CrossValid_params['test_size'],
                                                            CrossValid_params['min_period'], CrossValid_params['step']]
                                                            #CrossValid_params['gap']]
            cv_model = CrossValid(train_size = train_size, test_size = test_size, min_period = min_period)
            cv_dict = {'cv' : cv_model, 'step' : step, 'gap' : hor - 1}
            results = cross_valid(forecast_model, cv_dict, model, metrics) # (forecast_model, cv_dict, model, metrics)
            quality[tar]['forecast_model'].append(forecast_model)
            quality[tar]['train_loss'].append(results['train_loss'])
            quality[tar]['test_loss'].append(results['test_loss'])
            quality[tar]['mean_Y'].append(results['mean_Y'])
            quality[tar]['mean_Y_pred'].append(results['mean_Y_pred'])
            quality[tar]['mean_Y_train_pred'].append(results['mean_Y_train_pred'])
            quality[tar]['mean_Y_test'].append(results['mean_Y_test'])
            quality[tar]['importances'].append(results['importances'])
            quality[tar]['cv_dict'] = cv_dict
    return quality
            

def plot_cv_results(quality, horizons, plot_loss = False):
    for key in quality.keys():
        print(key)
        cv = quality[key]['cv_dict']['cv']
        train_index = list(range(cv.train_size))
        test_index = list(range(cv.train_size,cv.train_size + horizons))
        plt.plot(train_index,reduce(lambda a, b: a + b, quality[key]['mean_Y'])/horizons, label = 'mean_Y')
        plt.plot(train_index,reduce(lambda a, b: a + b, quality[key]['mean_Y_train_pred'])/horizons, alpha = .5, label = 'mean_Y_train_pred')
        plt.plot(test_index, np.reshape(quality[key]['mean_Y_test'], (horizons,1)),color = 'r', label = 'mean_Y_test')
        plt.plot(test_index, np.reshape(quality[key]['mean_Y_pred'],(horizons,1)), alpha = .65, label = 'mean_Y_pred')
        plt.legend()
        plt.show()
        if plot_loss:
            plt.plot(train_index, reduce(lambda a, b : a + b, quality[key]['train_loss']) / horizons, label = 'train_loss')
            plt.plot(test_index, np.reshape(quality[key]['test_loss'], (horizons, 1)), label = 'test_loss')
            plt.legend()
            plt.show()

def get_losses(quality, horizons):
    loss_dict = {}
    for key in quality.keys():
        cv = quality[key]['cv_dict']['cv']
        loss_dict[key] = {}
        train_index = list(range(cv.train_size))
        test_index = list(range(cv.train_size,cv.train_size + horizons))
        train_loss = pandas_Series(data = np.reshape(reduce(lambda a, b: a + b, quality[key]['train_loss']) / horizons, (len(train_index),)), index = train_index , name = 'train_loss')
        test_loss = pandas_Series(data =  np.reshape(quality[key]['test_loss'], (horizons,)), index = test_index, name = 'test_loss')
        loss_dict[key]['train_loss'], loss_dict[key]['test_loss'] = train_loss, test_loss
        return loss_dict
    
def get_importances(quality, horizons, plot_importances = False):
    importances_dict = {}
    for key in quality.keys():
        importances_dict[key] = []
        for hor in range(horizons):
            temp_series = pandas_Series(data = quality[key]['importances'][hor][0], index = quality[key]['forecast_model'][hor].features)
            importances_dict[key].append(temp_series)
            if plot_importances:
                plt.plot(list(temp_series.index), temp_series, color = 'b', label = key + '_hor_{}'.format(hor + 1))
                plt.legend()
                plt.show()
            
    return importances_dict