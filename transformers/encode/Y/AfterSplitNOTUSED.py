from sklearn.preprocessing import LabelEncoder


from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.encode.X.BeforeSplit import EncoderBeforeSplit

###############################################
## NO USED because not migrated to DataSpace ##
###############################################
class Tr_CategoricalEncoderAfterSplit_Y(EncoderBeforeSplit):
    ''' Only for Y, not tested yet'''
    def __init__(self, cat_cols=[], drop_first=True):
        EncoderBeforeSplit.__init__(self, cat_cols=cat_cols)
        self.drop_first = drop_first
    def transform(self, df):
        self.separator()
        cols = self.cat_cols
        self.print("cat cols ", cols)
        # Warning on LabelEncoder "This transformer should be used to encode target values, i.e. y, and not the input X."
        # see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        self.separator()
        for c in self.cat_cols:
            labelencoder = LabelEncoder()
            self.data[c] = labelencoder.fit_transform(self.data[c])
        return df
