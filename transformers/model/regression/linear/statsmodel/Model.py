
import statsmodels.api as sm
import statsmodels.stats.api as sms

from deepspace.transformers.model.Abstract import Abstract as AbstractModel
from deepspace.DataSpace import DataSpace
from deepspace.base import Base

class OLS(AbstractModel):
    def __init__(self) : #, perfchecker : MyPerformanceChecker):
        Base.__init__(self, '=', 50)
        AbstractModel.__init__(self)
        self._model = None
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
    def init_ds(self, ds):
        ds.x_train = self.x_train
        ds.x_test = self.x_test
        ds._model = self.get_model()
    def get_model(self):
        return self.fitted_model
    def tune (self, ds):
        #not compatible with cross_val_score
        pass
    def create_model_fit(self):
        self.x_train = sm.add_constant(self.x_train)
        self.x_test = sm.add_constant(self.x_test)
        self._model = sm.OLS(self.y_train, self.x_train)
        # Fitting the model
        self.fitted_model = self._model.fit()
        #print('type = ', type(ds.fitted_model.params))
    def create_model(self):
        self.x_train = sm.add_constant(self.x_train)
        self.x_test = sm.add_constant(self.x_test)
        self._model = sm.OLS(self.y_train, self.x_train)
    def fit(self):
        self.fitted_model = self._model.fit()
    def predict_train(self):
        self.train_pred = self._model.predict(self.x_train)
    def predict_test(self):
        self.test_pred = self._model.predict(self.x_test)

