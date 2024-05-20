import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.transformers.model.Abstract import Abstract
from deepspace.base import Base
from deepspace.DataSpace import DataSpace

from deepspace.Initialize import Initialize

class Model(Abstract):
    cv = 3
    scoring = 'neg_mean_squared_error'
    verbose = 3
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
        Abstract.__init__(self)
        self._model = None
        self.seed = seed
        self.max_depth = max_depth
        self.criterion= criterion
        self.min_samples_leaf = min_samples_leaf
        self.params = params
        self.do_tuning = do_tuning
        self.tuning_params = tuning_params
        self.show_summary = show_summary
        self.text_mode = text_mode
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        # Decision Tree Regressor
        self.create_model_fit()
        self.init_ds(ds)
        print('shapes ', self.x_train.shape, self.x_test.shape)
        if self.show_summary:
            self.summary()
        if self.do_tuning:
            cv = self.tuning_params['cv'] if 'cv' in self.tuning_params else Model.cv
            scoring = self.tuning_params['scoring'] if 'scoring' in self.tuning_params else Model.scoring
            verbose = self.tuning_params['verbose'] if 'verbose' in self.tuning_params else Model.verbose
            self.print(self.tuning_params['param_grid'])
            self.print({'cv': cv, 'verbose': verbose, 'scoring': scoring})
            best_dt, best_params = self.tune(self.ds,  
                                            parameters=self.tuning_params['param_grid'], 
                                            scoring=scoring,
                                            cv=cv,
                                            verbose=verbose)
            self.print(best_dt)
            self.print(best_params)
            self._model = best_dt
            self._model.fit(self.x_train, self.y_train)
            self.ds._model = self._model
        self.init_ds(ds)

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
        dtree_best_estimator = gridCV.best_estimator_

        # Fit the best estimator to the data
        dtree_best_estimator.fit(ds.x_train, ds.y_train)
        return dtree_best_estimator, gridCV.best_params_

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
