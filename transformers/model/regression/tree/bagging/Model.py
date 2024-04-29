from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from deepspace.transformers.model.regression.tree.Model import Model as DecisionTree
from deepspace.base import Base
from deepspace.DataSpace import DataSpace
from deepspace.Initialize import Initialize

class Model(DecisionTree):
    def __init__(self, seed=Initialize.seed) :
        Base.__init__(self, '=', 50)
        DecisionTree.__init__(self, seed=seed)
    def create_model_fit(self):
        self._model = BaggingRegressor(random_state = self.seed)
        # Fitting the model
        self._model.fit(self.x_train, self.y_train)
    def init_ds(self, ds):
        ds.model = self._model

