import itertools  # cartesian product
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.exploration.plot.Abstract import AbstractPlot


class ViolinPlot(AbstractPlot):
    def __init__(self, x, y, log_scale=False, xlabel=None, ylabel=None, figsize=(12, 7), color='red', xticks_rotation=None):
        AbstractPlot.__init__(self, x, y, xlabel=xlabel, ylabel=ylabel, figsize=figsize, color=color, xticks_rotation=xticks_rotation)
        self.log_scale = log_scale
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(ds.data, self.x, self.y, self.figsize, self.log_scale)
        return ds
    def plot(self, df, x, y, figsize=(12, 7), log_scale=False):
        self.separator(n=1, caller=self, string=f'plotting for {x}/{y}')
        sns.violinplot(data=df, x=x, y=y, log_scale=log_scale)
        plt.xticks(rotation=self.xticks_rotation)
        plt.show()
