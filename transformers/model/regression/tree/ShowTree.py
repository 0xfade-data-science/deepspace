from sklearn import tree
import matplotlib.pyplot as plt

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.base import Base

class ShowTree(Transformer):
    def __init__(self, max_depth=3, figsize=(25, 20), saveto=None) :
        Base.__init__(self, '=', 50)
        Transformer.__init__(self)
        self.max_depth=max_depth
        self.figsize=figsize
        self.saveto=saveto
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.show_tree_plot(max_depth=self.max_depth, figsize=self.figsize, saveto=self.saveto)
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds._model
        self.ds = ds
    def show_tree_plot(self, max_depth=3, figsize=(25, 20), saveto=None):
        self._show_tree_plot(self._model, max_depth=max_depth, figsize=figsize, saveto=saveto )
    def _show_tree_plot(self, model, max_depth=3, figsize=(25, 20), saveto=None):
        fig = plt.figure(figsize=figsize)

        target = self.ds.target_col
        if not type(self.ds.target_col) == list:
            target = [self.ds.target_col]

        box_representation = tree.plot_tree(model,
                              max_depth=max_depth,
                              feature_names=self.ds.x_train.columns.values.tolist(),
                              class_names=[],
                              filled=True)
        if saveto:
            plt.savefig(saveto)
