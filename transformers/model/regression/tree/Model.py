import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from DeepSpace.transformers.model.Abstract import Abstract
from DeepSpace.base import Base
from DeepSpace.DataSpace import DataSpace

from DeepSpace.Initialize import Initialize

class Model(Abstract):
    def __init__(self, seed=Initialize.seed, max_depth=None, criterion="squared_error", min_samples_leaf=1, show_summary=False, text_mode=True) :
        Base.__init__(self, '=', 50)
        Abstract.__init__(self)
        self._model = None
        self.seed = seed
        self.max_depth = max_depth
        self.criterion= criterion
        self.min_samples_leaf = min_samples_leaf
        self.show_summary = show_summary
        self.text_mode = text_mode
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        # Decision Tree Regressor
        self.create_model_fit()
        print('shapes ', self.x_train.shape, self.x_test.shape)
        self.init_ds(ds)
        if self.show_summary:
            self.summary()
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self.ds = ds
    def init_ds(self, ds):
        ds._model = self._model
    def get_model(self):
        return self._model
    def create_model_fit(self):
        self._model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            criterion=self.criterion,
            min_samples_leaf=self.min_samples_leaf,
            random_state = self.seed)
        # Fitting the model
        self._model.fit(self.x_train, self.y_train)
    def tune(self, ds, parameters={'max_depth': np.arange(2, 7),
                      'criterion': ['absolute_error', 'squared_error'],
                      'min_samples_leaf': [5, 10, 20, 25]
                    }, scoring='neg_mean_squared_error', cv=3, verbose=3):
        gridCV = GridSearchCV(ds._model, param_grid=parameters, scoring=scoring, cv=cv, verbose=verbose)

        # Fitting the grid search on the train data
        gridCV = gridCV.fit(ds.x_train, ds.y_train)

        # Set the classifier to the best combination of parameters
        dtree_estimator = gridCV.best_estimator_

        # Fit the best estimator to the data
        dtree_estimator.fit(ds.x_train, ds.y_train)
        return dtree_estimator

    def summary(self):
        self.show_tree(self.text_mode)
    def show_tree(self, text=True):
        if text:
            self.show_tree_text()
        else:
            self.show_tree_boxes()
    def show_tree_text(self):
        text_representation = tree.export_text(self._model)
        print(text_representation)
    def show_tree_plot(self, max_depth=3, figsize=(25, 20), saveto=None):
        fig = plt.figure(figsize=figsize)
        box_representation = tree.plot_tree(self._model,
                              max_depth=max_depth,
                              feature_names=self.ds.x_train.columns.values.tolist(),
                              class_names=self.ds.target_col,
                              filled=True)
        if saveto:
            plt.savefig(saveto)
    def show_feature_importance(self, max_depth=5, figsize = (4, 8), saveto=None):
        importances = self._model.feature_importances_
        columns = self.ds.x_train.columns
        importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False).head(max_depth)
        plt.figure(figsize = figsize)
        sns.barplot(x=importance_df.Importance, y=importance_df.index)
        if saveto:
            plt.savefig(saveto)
