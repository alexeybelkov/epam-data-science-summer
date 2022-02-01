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
