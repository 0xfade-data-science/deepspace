
from deepspace.transformers.model.regression.tree.ShowTree import ShowTree as BaseTreeViewer
from deepspace.base import Base
from deepspace.DataSpace import DataSpace

class TreeViewer(BaseTreeViewer):
    def __init__(self, max_depth=3, figsize=(25, 20), saveto=None) :
        Base.__init__(self, '=', 50)
        BaseTreeViewer.__init__(self, max_depth=max_depth, figsize=figsize, saveto=saveto)
    def show_tree_plot(self, max_depth=3, figsize=(25, 20), saveto=None):
        #we take the fist one
        self._show_tree_plot(self._model.estimators_[1], max_depth=max_depth, figsize=figsize, saveto=saveto )

