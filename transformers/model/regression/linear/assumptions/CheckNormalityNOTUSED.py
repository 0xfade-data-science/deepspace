from deepspace.DataSpace import DataSpace

class CheckNormality(Transformer):
    '''TODO this is graphical replace by Shapiro-Wilk Test'''
    def __init__(self, col):
        Transformer.__init__(self)
        self.col = col
    def transform(self, ds:DataSpace):
        self._check_normality(ds)
        return ds
    def _check_normality(self, ds):#compare distribution to normal distrib
        # Plot q-q plot of residuals
        #plt.close()
        data = ds.data[self.col]
        stats.probplot(data, dist = "norm", plot = pylab)
        plt.show()