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