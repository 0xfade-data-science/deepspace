########################################################################
import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class CheckResidualsMean(Transformer):
    def __init__(self, residuals_mean_threshold=0.0001):
        Transformer.__init__(self)
        self.residuals_mean_threshold = residuals_mean_threshold
    def transform(self, ds:DataSpace):
        self._model = ds._model
        val, bool = self._check_residuals_mean()
        ds.assumptions['residuals_mean_close_to_0'] = { 'value': val, 'bool': bool }
        return ds
    def _check_residuals_mean(self):
        residuals = self._model.resid  ## Complete the code
        val = abs(np.mean(residuals))
        bool = val < self.residuals_mean_threshold
        holds_or_not = "holds" if bool else "does not hold"
        self.print(f"residuals mean close to 0: assumption #1 {holds_or_not} ({val})" )
        return val, bool