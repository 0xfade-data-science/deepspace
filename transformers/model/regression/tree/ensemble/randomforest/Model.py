import numpy as np

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from deepspace.transformers.model.regression.tree.Model import Model as DecisionTree
from deepspace.base import Base
from deepspace.DataSpace import DataSpace
from deepspace.Initialize import Initialize

class Model(DecisionTree):
    def __init__(self, seed=Initialize.seed, 
                max_depth=None, criterion="squared_error", min_samples_leaf=1, 
                params = {'max_depth': None, 'criterion':"squared_error", 'min_samples_leaf':1},
                do_tuning=False, tuning_params={'param_grid':
                                        {'max_depth': np.arange(2, 7),
                                        'criterion': ['absolute_error', 'squared_error'],
                                        'min_samples_leaf': [5, 10, 20, 25]
                                        }, 'cv':3, 'scoring':'neg_mean_squared_error', 'verbose':3},
                show_summary=False, text_mode=True) :
        Base.__init__(self, '=', 50)
        DecisionTree.__init__(self, seed=seed, 
                                max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, 
                                params=params,
                                do_tuning=do_tuning, tuning_params=tuning_params,
                                show_summary=False, text_mode=True)
    def create_model_fit(self):
        self._model = RandomForestRegressor(
                    max_depth=self.max_depth,
                    criterion=self.criterion,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state = self.seed)
        # Fitting the model
        self._model.fit(self.x_train, self.y_train)
    def init_ds(self, ds):
        ds._model = self._model
