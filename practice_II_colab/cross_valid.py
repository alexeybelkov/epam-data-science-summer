class CrossValidOld:
    def __init__(self, n_samples, train_size=None, test_size=None, n_splits=None):
        self.n_samples = n_samples
        self.train_size = train_size
        self.test_size = test_size

    def split(self, type_, begin = 0, gap = 0, step = 1):
        
        train_begin, train_end = begin, begin + self.train_size
        test_begin = train_end + gap
        test_end = test_begin + self.test_size

        while test_end <= self.n_samples:
            yield range(train_begin, train_end), range(test_begin, test_end)
            train_begin += step if type_ == 'rolling' else 0
            train_end += step
            test_begin += step
            test_end += step

class CrossValid:
    def __init__(self, data, train_begin, train_end, test_end, dtype : 'ndarray or DataFrame'):
        self.train_begin = train_begin
        self.train_end = train_end
        self.test_end = test_end
        self.data = data
        self.dtype = dtype
    
    def split(self):
        for treshold in range(self.train_begin + 1, self.train_end):
            if self.dtype == 'ndarray':
                yield [self.data[(self.train_begin <= self.data[:,0]) & (self.data[:,0] < treshold)],
                       self.data[treshold <= self.data[:,0]]]
            elif self.dtype == 'DataFrame':
                yield [self.data.loc[range(self.train_begin,treshold)], 
                   self.data.loc[range(treshold,treshold+1)]]