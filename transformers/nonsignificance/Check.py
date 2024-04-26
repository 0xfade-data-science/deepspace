import pandas as pd

from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.Transformer import Transformer

class Check(Transformer):
    '''TODO'''
    def __init__(self, p_value_threshold = 0.05):
        Transformer.__init__(self)
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
    def check_non_significant_params_p_value(self):
        self.separator()
        removed = []
        debug = False
        pvalues = self._get_pvalues()
        for row in pvalues.iterrows():
            if debug:
                pdb.set_trace()
            rowid = row[0]
            col = row[1][0]
            coef = row[1][1]
            if abs(coef) > self.p_value_threshold:
                self.print(f"*{col}* should be removed because of coef greater than threshhold ({coef})")
                removed.append(col)
        self.removed = removed
        remstrcols = ', '.join(removed)
        self.print(f'features to remove: *{remstrcols}*')
        fcols = self.x_train.columns.values.tolist()
        strcols = ', '.join(fcols)
        self.print(f'remaining features : *{strcols}*')
    def _get_params(self):
        coef = self.get_model().params
        return pd.DataFrame({'Feature' : coef.index, 'Coefs' : coef.values})