from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from DeepSpace.transformers.model.regression.tree.Model import Model as DecisionTree
from DeepSpace.base import Base
from DeepSpace.DataSpace import DataSpace
from DeepSpace.Initialize import Initialize

class Model(DecisionTree):
    def __init__(self, seed=Initialize.seed, max_depth=None, criterion="log_loss", min_samples_leaf=1, show_summary=False, text_mode=True) :
        Base.__init__(self, '=', 50)
        DecisionTree.__init__(self, seed=seed, max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, show_summary=False, text_mode=True)
    def create_model_fit(self):
        self._model = RandomForestRegressor(
                    max_depth=self.max_depth,
                    criterion=self.criterion,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state = self.seed)
        # Fitting the model
        self._model.fit(self.x_train, self.y_train)

