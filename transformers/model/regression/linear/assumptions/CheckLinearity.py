import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.Transformer import Transformer

class Tr_OLSLinearity(Transformer):
    '''TODO this is graphical replace by Shapiro-Wilk Test'''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self._model = ds._model
        self._check_linearity()
        ds.assumptions['TODO_linearity'] = { }
        return ds
    def _check_linearity(self):
        # Predicted values
        fitted = self._model.fittedvalues
        residuals = self._model.resid  ## Complete the code
        # sns.set_style("whitegrid")
        sns.residplot(x = fitted, y = residuals, color = "lightblue", lowess = True)  ## Complete the code
        plt.xlabel("Fitted Values")
        plt.ylabel("Residual")
        plt.title("Residual PLOT")
        plt.show()
        return