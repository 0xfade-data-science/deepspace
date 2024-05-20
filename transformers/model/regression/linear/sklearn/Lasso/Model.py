from sklearn.linear_model import Lasso

from deepspace.transformers.model.regression.linear.sklearn.Ridge.Model import Model as RidgeModel

class Model(RidgeModel):
    def __init__(self,  dotuning=True, params={'alpha':[0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1, 5,10,20]} ) :
        RidgeModel.__init__(self, dotuning=dotuning, params=params )
    def init_ds(self, ds):
        ds._model = self._model
    def create_model_fit(self):
        self._model = Lasso(self.alpha)
        self._model.fit(self.x_train, self.y_train) # Fitting the data into the model



