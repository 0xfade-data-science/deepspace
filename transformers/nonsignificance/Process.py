import pandas as pd

from deepspace.DataSpace import DataSpace
from deepspace.transformers.drop.DropCols import DropCols

class RemoveNonSignificantFeatures(DropCols):
    '''TODO'''
    def __init__(self, p_value_threshold = 0.05):
        DropCols.__init__(self)
        self.p_value_threshold = p_value_threshold
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.remove_non_significant_params_p_value()
        self.init_ds(ds)
        ds.add_drop_cols(self.removed)
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self.ds = ds
        self._model = ds.get_model()
    def init_ds(self, ds):
        ds.x_train, ds.y_train, ds.x_test, ds.y_test = self.x_train, self.y_train, self.x_test, self.y_test
        ds.set_model(self._model)

    def _get_pvalues(self):
        pvalues = self.ds.get_model().pvalues
        return pd.DataFrame({'Feature' : pvalues.index, 'Coefs' : pvalues.values})
    def remove_non_significant_params_p_value(self):
        self.separator()
        removed = []
        debug = False
        pvalues = self._get_pvalues()
        for row in pvalues.iterrows():
            rowid = row[0]
            col = row[1][0]
            coef = row[1][1]
            if abs(coef) > self.p_value_threshold:
                self.print(f"Removing {col} beccause of coef greater than threshhold ({coef})")
                if col in self.x_train.columns.values.tolist() :
                    self.print(f"Droping {col}")
                    self.x_train.drop(col, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                    self.x_test.drop(col, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                removed.append(col)
        self.removed = removed
        remstrcols = ', '.join(removed)
        self.print(f'removed features : *{remstrcols}*')
        fcols = self.x_train.columns.values.tolist()
        strcols = ', '.join(fcols)
        self.print(f'remaining features : *{strcols}*')
    def _get_params(self):
        coef = self.get_model().params
        return pd.DataFrame({'Feature' : coef.index, 'Coefs' : coef.values})

class RemoveNonSignificantFeatures2(DropCols):
    '''TODO'''
    def __init__(self, p_value_threshold = 0.05):
        DropCols.__init__(self)
        self.p_value_threshold = p_value_threshold
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.remove_non_significant_params_p_value()
        self.init_ds(ds)
        ds.add_drop_cols(self.removed)
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self.ds = ds
        self._model = ds.get_model()
    def init_ds(self, ds):
        ds.x_train, ds.y_train, ds.x_test, ds.y_test = self.x_train, self.y_train, self.x_test, self.y_test
        ds.set_model(self._model)

    def _get_pvalues(self):
        pvalues = self.ds.get_model().pvalues
        return pd.DataFrame({'Feature' : pvalues.index, 'Coefs' : pvalues.values})
    def remove_non_significant_params_p_value(self):
        self.separator()
        removed = []
        debug = False
        pvalues = self._get_pvalues()

        self.ds.rnsf_x_train_removed = pd.DataFrame()
        self.ds.rnsf_x_test_removed = pd.DataFrame()

        for row in pvalues.iterrows():
            rowid = row[0]
            col = row[1][0]
            coef = row[1][1]
            if abs(coef) > self.p_value_threshold:
                self.print(f"Removing {col} beccause of coef greater than threshhold ({coef})")
                if col in self.x_train.columns.values.tolist() :
                    self.print(f"Removing {col}")
                    self.ds.rnsf_x_train_removed[col] = self.x_train[col]
                    self.ds.rnsf_x_test_removed[col] = self.x_test[col]
                    remaining_cols = self.x_train.columns.difference([col])
                    #self.x_train.drop(col, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                    self.x_train = self.x_train[remaining_cols]
                    #self.x_test.drop(col, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                    self.x_test = self.x_test[remaining_cols]
                removed.append(col)
        self.removed = removed
        remstrcols = ', '.join(removed)
        self.print(f'removed features : *{remstrcols}*')
        fcols = self.x_train.columns.values.tolist()
        strcols = ', '.join(fcols)
        self.print(f'remaining features : *{strcols}*')
    def _get_params(self):
        coef = self.get_model().params
        return pd.DataFrame({'Feature' : coef.index, 'Coefs' : coef.values})