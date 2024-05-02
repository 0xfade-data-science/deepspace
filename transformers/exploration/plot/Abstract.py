import itertools  # cartesian product
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer

class AbstractPlot(Transformer):
    def __init__(self, x, y, xlabel=None, ylabel=None, figsize=(12, 7), color='red', hue=None):
        Transformer.__init__(self)
        self.x = x
        self.y = y
        self.xlabel = xlabel if xlabel else x
        self.ylabel = ylabel if ylabel else y
        self.figsize = figsize
        self.color = color
        self.hue = hue

    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(self.x, self.y, self.figsize)
        return ds
    def plot(self, x, y, figsize):
        pass


