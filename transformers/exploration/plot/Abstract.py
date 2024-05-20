import itertools  # cartesian product
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.Abstract import Abstract as ColumnAbstract

class AbstractPlot(ColumnAbstract):
    def __init__(self, x, y, target_col=None, cat_cols=[], num_cols=[], exclude=[], only=[], 
                    xlabel=None, ylabel=None, figsize=(12, 7), color='red', xticks_rotation=None, hue=None):
        ColumnAbstract.__init__(self, target_col=target_col, cat_cols=cat_cols, num_cols=num_cols,
                    exclude=exclude, only=only)
        self.x = x
        self.y = y
        self.xlabel = xlabel if xlabel else x
        self.ylabel = ylabel if ylabel else y
        self.figsize = figsize
        self.color = color
        self.xticks_rotation = xticks_rotation
        self.hue = hue

    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(self.x, self.y, self.figsize)
        return ds
    def plot(self, x, y, figsize):
        pass


