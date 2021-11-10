from numpy import arange
from sklearn.model_selection import TimeSeriesSplit

class CV_split:
    def __init__(self, data_len, train_size, test_size):
        self.data_len = data_len
        self.train_size = train_size
        self.test_size = test_size
        self.folds = None

    def split(self, begin = 0, gap = 0, step = 1):
        
        train_begin, train_end = begin, begin + self.train_size
        test_begin = train_end + gap
        test_end = test_begin + self.test_size

        while test_end <= self.data_len:
            yield arange(train_begin, train_end), arange(test_begin, test_end)
            train_begin += step
            train_end += step
            test_begin += step
            test_end += step

    def k_fold(self, k, gap = 0):
        self.folds = TimeSeriesSplit(k, self.train_size. self.test_size, gap)