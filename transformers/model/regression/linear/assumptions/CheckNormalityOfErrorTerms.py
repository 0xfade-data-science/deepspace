import scipy.stats as stats
import matplotlib.pyplot as plt

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
    
class CheckNormalityOfErrorTerms(Transformer):
    '''TODO this is graphical replace by Shapiro-Wilk Test'''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self._model = ds._model
        self._check_normality_of_error_terms()
        ds.assumptions['TODO_normality_of_error_terms'] = { }
        return ds
    def _check_normality_of_error_terms(self):#compare distribution to normal distrib
        # Plot q-q plot of residuals
        residuals = self._model.resid  ## Complete the code
        stats.probplot(residuals, dist = "norm", plot = pylab)
        plt.show()

