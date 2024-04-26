import sklearn
from DeepSpace.transformers.model.Abstract import Abstract as AbstractModel
from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base

class LogisticRegression(AbstractModel):
    def __init__(self) :
        Base.__init__(self, '=', 50)
        AbstractModel.__init__(self)
    def get_model(self):
        return self._model
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
    def init_ds(self, ds):
        ds._model = self.get_model()
    def create_model_fit(self):
        self._model = sklearn.linear_model.LogisticRegression()
        # Fit the model on the training data
        self._model.fit(self.x_train, self.y_train)




