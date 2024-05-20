
########################################################################
######### Base class ###################################################
########################################################################


from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.base import Base
from deepspace.Initialize import Initialize

class Abstract(Transformer):
    def __init__(self, seed=Initialize.seed) : #, perfchecker : MyPerformanceChecker):
        Base.__init__(self, '=', 50)
        Transformer.__init__(self)
        self.seed = seed
        self._model = None
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.create_model_fit()
        self.predict()
        self.init_ds(ds)
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
    def init_ds(self, ds):
        ds._model = self.get_model()
    def get_model(self):
        return self._model
    def create_model_fit(self):
        pass
    def predict(self):
        self._model.predict(x)
    def fit(self):
        pass
    def tune(self):
        pass