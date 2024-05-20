from deepspace.transformers.model.regression.linear.statsmodel.Equation as LogModelEquation 

class LogLinRegModelEquation(LogModelEquation):
    def get_model(self, ds):
        return ds._model
    def get_coefs(self, ds):
        return self._model.coef_