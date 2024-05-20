from deepspace.transformers.model.regression.linear.Equation import Equation as BaseModelEquation

class Equation(BaseModelEquation):
    def __init__(self, target_col) : #, perfchecker : MyPerformanceChecker):
        BaseModelEquation.__init__(self, target_col)
    def get_model(self, ds):
        return ds._model

class LogModelEquation(Equation):
    def equation(self):
        y, X = self._equation()
        equation = f"log({y}) = " + '\n\t+ '.join(X)
        print(equation)

