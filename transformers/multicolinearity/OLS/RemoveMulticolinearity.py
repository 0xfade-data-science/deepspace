import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor

from deepspace.DataSpace import DataSpace
from deepspace.transformers.drop.DropColsXYXY import DropColsXYXY
from deepspace.transformers.column.Abstract import Abstract

class RemoveMulticolinearity(DropColsXYXY, Abstract):
    '''TODO'''
    def __init__(self, num_cols=[]):
        DropColsXYXY.__init__(self)
        Abstract.__init__(self, num_cols=num_cols)
    def transform(self, ds:DataSpace):
        self.ds = ds
        x_train, y_train, x_test, y_test = self.remove_multicolinearity(ds.x_train, ds.y_train, ds.x_test, ds.y_test)
        ds.x_train, ds.y_train = x_train, y_train
        ds.x_test, ds.y_test = x_test, y_test
        #x_test, y_test = self.remove_multicolinearity(ds.x_test, ds.y_test)
        #ds.x_test, ds.y_test = x_test, y_test
        ds.add_drop_cols(self.removed)
        return ds
    def remove_multicolinearity(self, x_train, y_train, x_test, y_test):
        self.separator()
        xtrain, y_train, x_test, y_test = self.remove_multicolinearity_num(x_train, y_train, x_test, y_test)
        return xtrain, y_train, x_test, y_test
    def remove_multicolinearity_num(self, x_train, y_train, x_test, y_test, seen=[], removed=[]):
        self.separator()
        #pdb.set_trace()
        cols = self._get_num_cols(self.ds) # we can use self.num_cols instead of cols
        vif = self._check_vif(x_train).drop(0, axis=0) #remove const line
        if len(seen): #avoid checking again those already processed
            for c in seen:
                idx = vif.query(f'feature == "{c}"').index
                vif.drop(idx, inplace=True, axis=0)
        vif['VIF'] = vif['VIF'].astype(float)
        vif.reset_index(inplace=True, drop=True)
        rowmax = vif['VIF'].idxmax()
        colmax = vif.loc[rowmax][0]
        valmax = vif.loc[rowmax][1]
        self.print(f"Processing {colmax}(vif = {valmax})...")
        if valmax > 5 :
            self.print(f"Processing {colmax} because of valmax {valmax}")
            if colmax not in seen :
                if colmax in cols: #cat_cols must be processed through p-value not here
                    #pdb.set_trace()
                    removed.append(colmax)
                    #pdb.set_trace()
                    if colmax in x_train.columns.values.tolist():
                        self.print(f"Removing {colmax}")
                        x_train.drop(colmax, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                        x_test.drop(colmax, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                        #self.ds.add_drop_cols(removed)
                        #self.ds = Tr_DropColsXYXY.transform(self, ds)
                    else:
                        self.print(f"Col {colmax} not in dataframe columns")

                else:
                    self.print(f"Rejecting  categorical feature {colmax}")

                seen.append(colmax)
            else:
                self.print(f"Already seen {colmax}")

            x_train, y_train, x_test, y_test = self.remove_multicolinearity_num(x_train, y_train, x_test, y_test, seen=seen, removed=removed)
        #vif = self._check_vif(x_train).drop(0, axis=0) #remove const line
        #self.print(vif)
        strrmv = ', '.join(removed)
        self.print(f'removed cols : *{strrmv}*')
        remcols = x_train.columns.values.tolist()
        remstr = ', '.join(remcols)
        self.print(f'remaining cols  : *{remstr}*')
        self.ds.colin_rem_cols = self.removed = removed
        return x_train, y_train, x_test, y_test

    def _check_vif(self, train):
        vif = pd.DataFrame()
        vif["feature"] = train.columns
        # Calculating VIF for each feature
        vif["VIF"] = [
            variance_inflation_factor(train.values, i) for i in range(len(train.columns))
        ]
        return vif

class RemoveMulticolinearity2(DropColsXYXY, Abstract):
    '''TODO'''
    def __init__(self, num_cols=[]):
        DropColsXYXY.__init__(self)
        Abstract.__init__(self, num_cols=num_cols)
    def transform(self, ds:DataSpace):
        self.ds = ds
        x_train, y_train, x_test, y_test, x_train_removed, x_test_removed = self.remove_multicolinearity(ds.x_train, ds.y_train, ds.x_test, ds.y_test)
        ds.x_train, ds.y_train = x_train, y_train
        ds.x_test, ds.y_test = x_test, y_test
        ds.rmc_x_train_removed, ds.rmc_x_test_removed = x_train_removed, x_test_removed
        #x_test, y_test = self.remove_multicolinearity(ds.x_test, ds.y_test)
        #ds.x_test, ds.y_test = x_test, y_test
        ds.add_drop_cols(self.removed)
        return ds
    def remove_multicolinearity(self, x_train, y_train, x_test, y_test):
        self.separator()
        return self.remove_multicolinearity_num(x_train, y_train, x_test, y_test, pd.DataFrame(), pd.DataFrame())
    def remove_multicolinearity_num(self, x_train, y_train, x_test, y_test, x_train_removed, x_test_removed, seen=[], removed=[]):
        self.separator()
        #pdb.set_trace()
        cols = self._get_num_cols(self.ds) # we can use self.num_cols instead of cols
        vif = self._check_vif(x_train).drop(0, axis=0) #remove const line
        if len(seen): #avoid checking again those already processed
            for c in seen:
                idx = vif.query(f'feature == "{c}"').index
                vif.drop(idx, inplace=True, axis=0)
        vif['VIF'] = vif['VIF'].astype(float)
        vif.reset_index(inplace=True, drop=True)
        rowmax = vif['VIF'].idxmax()
        colmax = vif.loc[rowmax][0]
        valmax = vif.loc[rowmax][1]
        self.print(f"Processing {colmax}(vif = {valmax})...")
        if valmax > 5 :
            self.print(f"Processing {colmax} because of valmax {valmax}")
            if colmax not in seen :
                if colmax in cols: #cat_cols must be processed through p-value not here
                    #pdb.set_trace()
                    removed.append(colmax)
                    #pdb.set_trace()
                    if colmax in x_train.columns.values.tolist():
                        self.print(f"Removing {colmax}")
                        x_train_removed[colmax] = x_train[colmax]
                        x_test_removed[colmax] = x_test[colmax]
                        remaining_cols = x_train.columns.difference([colmax])
                        #x_train.drop(colmax, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                        x_train = x_train[ remaining_cols ]
                        #x_test.drop(colmax, inplace=True, axis=1) # maybe need to remove the var from num_cols and cat_cols and/or add to drop_cols
                        x_test = x_test[ remaining_cols ]
                        #self.ds.add_drop_cols(removed)
                        #self.ds = Tr_DropColsXYXY.transform(self, ds)
                    else:
                        self.print(f"Col {colmax} not in dataframe columns")

                else:
                    self.print(f"Rejecting  categorical feature {colmax}")

                seen.append(colmax)
            else:
                self.print(f"Already seen {colmax}")

            x_train, y_train, x_test, y_test, x_train_removed, x_test_removed = self.remove_multicolinearity_num(x_train, y_train, x_test, y_test, x_train_removed, x_test_removed, seen=seen, removed=removed)
        #vif = self._check_vif(x_train).drop(0, axis=0) #remove const line
        #self.print(vif)
        strrmv = ', '.join(removed)
        self.print(f'removed cols : *{strrmv}*')
        remcols = x_train.columns.values.tolist()
        remstr = ', '.join(remcols)
        self.print(f'remaining cols  : *{remstr}*')
        self.ds.colin_rem_cols = self.removed = removed
        return x_train, y_train, x_test, y_test, x_train_removed, x_test_removed

    def _check_vif(self, train):
        vif = pd.DataFrame()
        vif["feature"] = train.columns
        # Calculating VIF for each feature
        vif["VIF"] = [
            variance_inflation_factor(train.values, i) for i in range(len(train.columns))
        ]
        return vif
