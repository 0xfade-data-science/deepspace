
from DeepSpace.transformers.Meta import Meta
from DeepSpace.transformers.OLS.assumptions.CheckNormalityOfErrorTerms import CheckNormalityOfErrorTerms
from DeepSpace.transformers.OLS.assumptions.CheckNormalityOfErrorTerms import CheckResidualsMean
from DeepSpace.transformers.OLS.assumptions.CheckNormalityOfErrorTerms import CheckHomoscedasticity
from DeepSpace.transformers.OLS.assumptions.CheckNormalityOfErrorTerms import Linearity
from DeepSpace.DataSpace import DataSpace

class CheckALLAssumptionsOnResiduals(Meta):
    '''TODO'''
    def __init__(self, transformers=[]):
        Meta.__init__(self, transformers=transformers)
    def transform(self, ds:DataSpace):
        return self.check_regression_assumptions(ds)
    def check_regression_assumptions(self, ds:DataSpace):
        #source : Learners_Notebook_Boston_House_Price_Prediction_LowCode.ipynb
        #Mean of residuals should be 0
        assumptions = [
            self._check_residuals_mean(ds)
            #No Heteroscedasticity
            , self._check_heteroscedasticity(ds)
            #Linearity of variables
            , self._check_linearity(ds)
            #Normality of error terms
            , self._check_normality_of_error_terms(ds)
          ]
        def isTrue(a,b): return a and b
        return reduce(isTrue, assumptions)
    def _check_residuals_mean(self):
        ...