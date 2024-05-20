from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class Equation(Transformer):
    def __init__(self, target_col=None) : 
        Transformer.__init__(self)
        self.target_col = target_col
    def transform(self, ds:DataSpace):
        if not self.target_col:
            self.target_col = ds._get_target_col()
        self._model = self.get_model(ds)
        self.coefs = self.get_coefs(ds)
        self.equation()
        return ds
    def equation(self):
        y, X = self._equation()
        equation = f"{y} = " + '\n\t+ '.join(X)
        print(equation)
    def _equation(self):
        coef = self.coefs
        coef = coef.sort_values(ascending=False, key=lambda a: abs(a))
        target_variable = self.target_col
        return target_variable, [ f'({coef[i]}) * {coef.index[i]}' for i in range(len(coef)) ]
    def get_model(self, ds):
        return ds._model
    def get_coefs(self, ds):
        return self._model.params

