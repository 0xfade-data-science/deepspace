from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.base import Base
from DeepSpace.DataSpace import DataSpace

class Show(Transformer):
    def __init__(self, text=True) : #, perfchecker : MyPerformanceChecker):
        Base.__init__(self, '=', 50)
        Transformer.__init__(self)
        self.text = text
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.show_tree(self.text)
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds.model
    def show_tree(self, text=True):
        if text:
            self.show_tree_text()
        else:
            self.show_tree_boxes()
    def show_tree_boxes(self):
        fig = plt.figure(figsize=(25, 20))
        box_representation = tree.plot_tree(self._model,
                              feature_names=self.ds.x_train.columns.values.tolist(),
                              class_names=self.ds.target_col,
                              filled=True)

    def show_tree_text(self):
        text_representation = tree.export_text(self._model)
        print(text_representation)