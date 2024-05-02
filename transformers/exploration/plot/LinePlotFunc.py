import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.exploration.plot.LinePlot import LinePlot

class LinePlotFunc(LinePlot):
    def __init__(self, x, y, func=None, estimator='mean', errorbar=None, xlabel=None, ylabel=None, figsize=(12, 7), color='red', hue=None):
        LinePlot.__init__(self, x, y, estimator=estimator, errorbar=errorbar, xlabel=xlabel, ylabel=ylabel, figsize=figsize, color=color, hue=hue)
        self.func = func
    def transform(self, ds:DataSpace):
        if self.func :
            y = f'func_{self.y}'
            ds.data[y] = self.func(ds.data[self.y])
            self.y = y
        LinePlot.transform(self, ds)
        return ds
