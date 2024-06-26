import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.exploration.plot.Abstract import AbstractPlot

class BoxPlot(AbstractPlot):
    def __init__(self, x, y, xlabel=None, ylabel=None, figsize=(12, 7), color='red', xticks_rotation=90):
        AbstractPlot.__init__(self, x, y, xlabel=xlabel, ylabel=ylabel, figsize=figsize, color=color, xticks_rotation=xticks_rotation)
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(ds.data, self.x, self.y)
        return ds
    def plot(self, data, x, y):
        self.separator(n=1, caller=self, string=f'plotting for {x}/{y}')

        plt.figure(figsize = self.figsize)  
        sns.boxplot(x=self.x, y=self.y, data=data)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        plt.xticks(rotation=self.xticks_rotation)
        plt.show()     

