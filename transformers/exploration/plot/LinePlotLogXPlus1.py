import numpy as np

from deepspace.transformers.exploration.plot.LinePlotFunc import LinePlotFunc

class LinePlotLogXPlus1(LinePlotFunc):
    def __init__(self, x, y, estimator='mean', errorbar=None, xlabel=None, ylabel=None, figsize=(12, 7), color='red', hue=None):
        LinePlotFunc.__init__(self, x, y, func=np.log1p, estimator=estimator, errorbar=errorbar, xlabel=xlabel, ylabel=ylabel, figsize=figsize, color=color, hue=hue)
