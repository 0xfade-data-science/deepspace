from deepspace.transformers.separate.Separator import Separator
from deepspace.DataSpace import DataSpace

class InverseSeparator(Separator):
    def __init__(self, target_col):
        Separator.__init__(self)
        if not target_col:
            raise Exception("target_col null")
        self.target_col = target_col
    def transform(self, ds: DataSpace):
        self.ds = ds
        self.sanity_check()
        ds = self.inverse_separate(ds)
        ds.isSeparateReverted_YTrain = True

        return ds
    def sanity_check(self):
        if not self.ds.isSeparateApplyed:
            raise Exception('not separated')

    def get_data(self):
        #no Y inverted before
        #TODO avoid copy by just renaming target_col to target_col_pred
        #if ds.isRmcUnApplyed_XTrain or ds.isRnsfUnApplyed_XTrain:
        if hasattr(self.ds, 'inv_train_data'):
            return self.ds.inv_train_data.copy()
        else:
            return self.ds.x_train.copy()
    def inverse_separate(self, ds: DataSpace):
        self.separator(caller=self)
        y_pred = ds._model.predict(ds.x_train)

        data = self.get_data()
        data[self.target_col] = y_pred

        if 'const' in data:
            data.drop(columns=['const'], inplace=True)

        #take only new columns but reordered
        ordered_cols = self._get_ordered_cols(ds.data.columns, data.columns)
        ds.inv_train_data_pred = data[ordered_cols]
        #==== FUTURE
        ds.inverted_ds.y_train_pred = y_pred 
        ds.inverted_ds.y_train_pred_all_cols = ds.inv_train_data_pred  
        
        return ds