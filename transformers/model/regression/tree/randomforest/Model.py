from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from deepspace.transformers.model.regression.tree.Model import Model as DecisionTree
from deepspace.base import Base
from deepspace.DataSpace import DataSpace
from deepspace.Initialize import Initialize

class Model(DecisionTree):
    def __init__(self, seed=Initialize.seed, max_depth=None, criterion="squared_error", min_samples_leaf=1, show_summary=False, text_mode=True) :
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
    def init_ds(self, ds):
        ds._model = self._model
'''
class Tr_MyRandForestRegPerformanceTransformer(Tr_MyDecTreeRegPerformanceTransformer):
    def __init__(self) :
        MyBaseClass.__init__(self, '#!#', 50)
        Tr_MyDecTreeRegPerformanceTransformer.__init__(self)
    def transform_remove(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()
        return ds
    def from_ds_init_remove(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds.model
    def get_model_remove(self):
        return self._model
class Tr_ShowMyRandForestRegPerformance(Tr_ShowMyLinRegPerformance):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        pdf = pd.concat([ds.perf_train, ds.perf_test], keys=['train', 'test'], ignore_index=False)
        self.display(pdf)
        ds.perf_df = pdf
        return ds

class Tr_MyRandForestRegShowTree(Tr_MyDecTreeRegShowTree):
    def __init__(self, max_depth=3, figsize=(25, 20), saveto=None) :
        MyBaseClass.__init__(self, '=', 50)
        Tr_MyDecTreeRegShowTree.__init__(self, max_depth=max_depth, figsize=figsize, saveto=saveto)
    def show_tree_plot(self, max_depth=3, figsize=(25, 20), saveto=None):
        #we take the fist one
        self._show_tree_plot(myOptimRFModel._model.estimators_[1], max_depth=max_depth, figsize=figsize, saveto=saveto )

'''