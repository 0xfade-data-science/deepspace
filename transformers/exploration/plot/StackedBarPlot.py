import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.exploration.plot.Abstract import AbstractPlot

class StackedBarPlot(AbstractPlot):
    def __init__(self, x, y, xlabel=None, ylabel=None, figsize=None, color='red', xticks_rotation=90):
        AbstractPlot.__init__(self, x, y, xlabel=xlabel, ylabel=ylabel, figsize=figsize, color=color, xticks_rotation=xticks_rotation)
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(self.x, self.y)
        return ds
    def plot(self, x, y):
        self.separator(n=1, caller=self, string=f'plotting for {x}/{y}')

        """
        Print the category counts and plot a stacked bar chart
        x: independent variable
        y: independent variable
        """
        data = self.ds.data
        X = data[x]
        Y = data[y]
        count = X.nunique()
        sorter = Y.value_counts().index[-1]
        tab1 = pd.crosstab(X, Y, margins=True).sort_values(by=sorter, ascending=False)
        self.display(tab1)
        print("_" * 120)
        tab = pd.crosstab(X, Y, normalize="index").sort_values(by=sorter, ascending=False)
        figsize=(count + 1, 5)
        if self.figsize:
            figsize=self.figsize
        tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
        plt.legend(loc="lower left", frameon=False,)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.xticks(rotation=self.xticks_rotation)
        plt.show()
