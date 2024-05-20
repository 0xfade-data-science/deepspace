
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.chain.Milestone import Milestone
from deepspace.transformers.model.regression.linear.assumptions.CheckNormalityOfErrorTerms import CheckNormalityOfErrorTerms
from deepspace.transformers.model.regression.linear.assumptions.CheckResidualsMean import CheckResidualsMean
from deepspace.transformers.model.regression.linear.assumptions.CheckHomoscedasticity import CheckHomoscedasticity
from deepspace.transformers.model.regression.linear.assumptions.CheckLinearity import CheckLinearity
from deepspace.DataSpace import DataSpace

class CheckAll4RegressionAssumptions(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)

    def transform(self, ds:DataSpace):
        return self.check_regression_assumptions(ds)
    def check_regression_assumptions(self, ds:DataSpace):
        _ = ( Milestone(ds)
            >> CheckResidualsMean()
            >> CheckHomoscedasticity()
            >> CheckLinearity()
            >> CheckNormalityOfErrorTerms()
        )
        return _.ds
