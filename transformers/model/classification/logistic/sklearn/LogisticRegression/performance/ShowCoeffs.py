import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base
from DeepSpace.transformers.Transformer import Transformer


class ShowCoeffs(Transformer):
    def __init__(self):
        Base.__init__(self, '#!#', 50)
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.display(self.get_coeffs())
        return ds
    def from_ds_init(self, ds):
        self.ds = ds
        self._model = ds._model
    def get_coeffs(self):
        # Printing the coefficients of logistic regression
        cols = self.ds.x_train.columns
        coef_lg = self._model.coef_
        return pd.DataFrame(coef_lg,columns = cols).T.sort_values(by = 0, ascending = False)
    
class ShowInvCoeffs(ShowCoeffs):
    def __init__(self):
        Base.__init__(self, '#!#', 50)
        ShowCoeffs.__init__(self)
    def get_coeffs(self):
        # Printing the coefficients of logistic regression
        cols = self.ds.x_train.columns
        coef = self._model.coef_[0]
        odds = np.exp(coef) # Finding the odds
        return pd.DataFrame(odds, cols, columns = ['odds']).sort_values(by = 'odds', ascending = False)    