
import statsmodels.stats.api as sms

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

# the error does not change much as the preodictor changes ?
class CheckHomoscedasticity(Transformer):
    def __init__(self, p_value_threshold = 0.05):
        Transformer.__init__(self)
        self.p_value_threshold = p_value_threshold
    def transform(self, ds:DataSpace):
        self.x_train, self._model = ds.x_train, ds._model
        val, bool= self._check_homoscedasticity()
        ds.assumptions['no_heteroscedasticity'] = { 'value': val, 'bool': bool }
        return ds
    def _check_homoscedasticity(self):
        residuals = self._model.resid  ## Complete the code
        name = ["F statistic", "p-value"]
        test = sms.het_goldfeldquandt(residuals, self.x_train) ## Complete the code
        res = lzip(name, test)
        val = res[1][1]
        bool = val > self.p_value_threshold
        fail_or_not, holds_or_not = ("fail", "holds") if val else ("not fail", "does not hold")
        self.print(f"No heteroscedasticity: null hypotesis {fail_or_not} hence assumption #2 {holds_or_not} ({val})" )
        return val, bool
