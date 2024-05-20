import pandas as pd 
#from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.transformers.Transformer import Transformer
from deepspace.base import Base
from deepspace.DataSpace import DataSpace

class ShowMainFeatures(Transformer):
    def __init__(self, max_depth=3, figsize = (8, 1), saveto=None) :
        Base.__init__(self, '=', 50)
        Transformer.__init__(self)
        self.max_depth=max_depth
        self.figsize=figsize
        self.saveto=saveto
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.show_feature_importance(max_depth=self.max_depth, figsize=self.figsize, saveto=self.saveto)
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds._model
    def show_feature_importance(self, max_depth, figsize, saveto):
        importances = self._model.feature_importances_
        columns = self.ds.x_train.columns
        importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False).head(max_depth)
        plt.figure(figsize = figsize)
        sns.barplot(x=importance_df.Importance, y=importance_df.index)
        if saveto:
            plt.savefig(saveto)

class SaveMainFeatures(ShowMainFeatures):
    def __init__(self, saveto, max_depth=3, figsize=(8, 1)) :
        Base.__init__(self, '=', 50)
        ShowMainFeatures.__init__(self, saveto=saveto, max_depth=max_depth, figsize=figsize)
