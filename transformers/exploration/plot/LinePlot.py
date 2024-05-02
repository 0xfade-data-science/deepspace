import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.exploration.plot.Abstract import AbstractPlot

class LinePlot(AbstractPlot):
    def __init__(self, x, y, estimator='mean', errorbar=None, xlabel=None, ylabel=None, figsize=(12, 7), color='red', hue=None):
        AbstractPlot.__init__(self, x, y, xlabel=xlabel, ylabel=ylabel, figsize=figsize, color=color, hue=hue)
        self.estimator = estimator
        self.errorbar = errorbar
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(self.x, self.y, estimator=self.estimator, errorbar=self.errorbar, color=self.color, hue=self.hue)
        return ds
    def plot(self, x, y, estimator='mean', errorbar=None, color='red', hue=None):
        self.separator(n=1, caller=self, string=f'plot for {x}/{y}')
        data = self.ds.data
        #TODO
        #if x in self.ord_cols:
        #  sorter = data.sort_values(by=x, ascending=True).reset_index()
        #else:
        #  sorter = data[[x, y]].groupby([x]).mean().sort_values(by=y, ascending=False).reset_index()
        # lineplot 
        sns.lineplot(y=y, x=x, data=data, errorbar=self.errorbar, estimator=estimator, color=color, hue=hue)
        plt.xticks(rotation=90)
        plt.show()

