import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.exploration.plot.Abstract import AbstractPlot

class BarPlot(AbstractPlot):
    def __init__(self, x, y, xlabel=None, ylabel=None, figsize=(12, 7), color='red', xticks_rotation=90):
        AbstractPlot.__init__(self, x, y, xlabel=xlabel, ylabel=ylabel, figsize=figsize, color=color, xticks_rotation=xticks_rotation)
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(self.ds.data, self.x, self.y)
        return ds
    def plot(self, data, x, y):
        self.separator(n=1, caller=self, string=f'plotting for {x}/{y}')
        #TODO
        #if x in self.ord_cols:
        #  sorter = data.sort_values(by=x, ascending=True).reset_index()
        #else:
        #  sorter = data[[x, y]].groupby([x]).mean().sort_values(by=y, ascending=False).reset_index()
        # barplot shows only the mean
        #sns.barplot(y=y, x=x, data=data, order=sorter[x])
        sns.barplot(y=y, x=x, data=data)
        plt.xticks(rotation=self.xticks_rotation)
        plt.show()

