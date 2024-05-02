########################################################################
######### Basic Data Structures ########################################
########################################################################
import pandas as pd
import copy as copier

from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.column.abstract import Abstract as AbstractTransformer
from deepspace.transformers.file.File import File

class DataSpace(AbstractTransformer):
    def __init__(self, docopy=True):
        super().__init__()
        self.target_col = None
        self.model = None
        self.num_cols = []
        self.cat_cols = []
        self.id_cols = []
        self.drop_cols = []
        self.assumptions = {}
        self.data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    def transform(self, file:File):
        #self.filepath, self.filesep = specs
        #self.data = pd.read_csv(self.filepath, sep=self.filesep)
        self.data = file.extract()
        self.num_cols = self.get_num_cols()
        self.cat_cols = self.get_cat_cols()
        self.result = self.data
        return self
    def clone(self):
        #clone = MyDataSpace()
        copy = copier.deepcopy(self)
        assert(type(copy) == type(self))
        return copy
    def clean(self):
        del self.data
        self.data = None
    def get_data(self):
        return self.data
    def get_model(self):
        return self._model
    def set_model(self, model):
        self._model = model

    def get_target_col(self):
        return self.target_col
    def get_id_cols(self):
        return self.id_cols
    def get_num_cols(self):
        return self.num_cols
    def get_cat_cols(self):
        return self.cat_cols

    def set_target_col(self, target_col):
        self.target_col = target_col
    def set_id_cols(self, cols):
        self.id_cols = cols
    def set_num_cols(self, cols):
        self.num_cols = cols
    def set_cat_cols(self, cols):
        self.cat_cols = cols

    def set_drop_cols(self, cols):
        self.drop_cols = cols
    def get_drop_cols(self):
        return self.drop_cols
    def add_drop_col(self, col):
        self.drop_cols.append(col)
        new_cols = list(set(self.drop_cols))
        self.set_drop_cols(new_cols) #unique values
    def add_drop_cols(self, cols):
        _cols = self.drop_cols + cols
        new_cols = list(set(_cols))
        self.set_drop_cols(new_cols) #unique values

    def show_data(self, view=True):
        self.separator(caller=self)
        self.print(self.data.shape)
    def show_x(self, view=True):
        self.separator(caller=self)
        self.print(self.x.shape)
    def show_y(self, view=True):
        self.separator(caller=self)
        self.print(self.y.shape)
    def show_x_train(self, view=True):
        self.separator(caller=self)
        self.print(self.x.shape)
    def show_y_train(self, view=True):
        self.separator(caller=self)
        self.print(self.y.shape)
    def show_x_test(self, view=True):
        self.separator(caller=self)
        self.print(self.x.shape)
    def show_y_test(self, view=True):
        self.separator(caller=self)
        self.print(self.y.shape)
    def shape(self, view=True):
        self.separator(caller=self)
        return self.data.shape
    def show_shape(self):
        self.separator(caller=self)
        self.show_data()
        self.show_x()
        self.show_y()
        self.show_x_train()
        self.show_x_test()
        self.show_x_test()
        self.show_x_test()
    def head(self):
        self.separator()
        self.display(self.data.head())

    def info(self):
        self.separator()
        self.data.info()

    def describe(self):
        self.separator()
        self.display(self.data.describe().T)

    def overview(self):
        self.show_shape()
        self.head()
        self.info()
        self.describe()

    def pivot(self, index, columns, values):
        self.dfpv = self.data.pivot_table(index=index, columns=columns, values=values)
        return self.dfpv

class DataSpaceNew(AbstractTransformer):
    def __init__(self, docopy=True):
        super().__init__()
        self.target_col = None
        self.num_cols = []
        self.cat_cols = []
        self.id_cols = []
        self.drop_cols = []
        self.model = None
        self.assumptions = {}
        self.data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    def transform(self, file:File):
        #self.filepath, self.filesep = specs
        #self.data = pd.read_csv(self.filepath, sep=self.filesep)
        self.data = DataVector()
        df = file.extract()
        self.data.set_data(df)
        self.num_cols = self.get_num_cols()
        self.cat_cols = self.get_cat_cols()
        self.result = self.data
        return self
    def clone(self):
        #clone = MyDataSpace()
        copy = copier.deepcopy(self)
        assert(type(copy) == type(self))
        return copy
    def clean(self):
        del self.data
        self.data = None

    def get_model(self):
        return self._model
    def set_model(self, model):
        self._model = model

    def get_target_col(self):
        return self.target_col
    def get_id_cols(self):
        return self.id_cols
    def get_num_cols(self):
        return self.num_cols
    def get_cat_cols(self):
        return self.cat_cols

    def set_target_col(self, target_col):
        self.target_col = target_col
    def set_id_cols(self, cols):
        self.id_cols = cols
    def set_num_cols(self, cols):
        self.num_cols = cols
    def set_cat_cols(self, cols):
        self.cat_cols = cols

    def set_drop_cols(self, cols):
        self.drop_cols = cols
    def get_drop_cols(self):
        return self.drop_cols
    def add_drop_col(self, col):
        self.drop_cols.append(col)
        new_cols = list(set(self.drop_cols))
        self.set_drop_cols(new_cols) #unique values
    def add_drop_cols(self, cols):
        _cols = self.drop_cols + cols
        new_cols = list(set(_cols))
        self.set_drop_cols(new_cols) #unique values

    def show_data(self, view=True):
        self.separator(caller=self)
        self.print(self.data.shape)
    def show_x(self, view=True):
        self.separator(caller=self)
        self.print(self.x.shape)
    def show_y(self, view=True):
        self.separator(caller=self)
        self.print(self.y.shape)
    def show_x_train(self, view=True):
        self.separator(caller=self)
        self.print(self.x.shape)
    def show_y_train(self, view=True):
        self.separator(caller=self)
        self.print(self.y.shape)
    def show_x_test(self, view=True):
        self.separator(caller=self)
        self.print(self.x.shape)
    def show_y_test(self, view=True):
        self.separator(caller=self)
        self.print(self.y.shape)
    def shape(self, view=True):
        self.separator(caller=self)
        return self.data.shape
    def show_shape(self):
        self.separator(caller=self)
        self.show_data()
        self.show_x()
        self.show_y()
        self.show_x_train()
        self.show_x_test()
        self.show_x_test()
        self.show_x_test()
    def head(self):
        self.separator()
        self.display(self.data.head())

    def info(self):
        self.separator()
        self.data.info()

    def describe(self):
        self.separator()
        self.display(self.data.describe().T)

    def overview(self):
        self.show_shape()
        self.head()
        self.info()
        self.describe()

