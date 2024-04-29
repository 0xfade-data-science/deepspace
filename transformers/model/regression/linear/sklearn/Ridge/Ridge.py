from sklearn.linear_model import Ridge

from deepspace.Initialize import Initialize
from deepspace.DataSpace import DataSpace
from deepspace.base import Base
from deepspace.transformers.model.Abstract import Abstract as AbstractModel

class Model(AbstractModel):
    def __init__(self, seed=Initialize.seed, dotuning=True, params={'alpha':[0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1, 5,10,20]} ) :
        Base.__init__(self, '=', 50)
        AbstractModel.__init__(self, seed=seed)
        self._model = None
        self.dotuning = dotuning
        self.params = params
        self.alpha = None
    def transform(self, ds:DataSpace):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        # Fitting the model
        if self.dotuning:
            self.tune(ds, self.params)
            self.alpha = self.best_params['alpha']
        self.create_model_fit() #creating Ridge Regression model
        self.init_ds(ds)
        return ds
    def init_ds(self, ds):
        ds._model = self._model
    def create_model(self):
        self._model = Ridge(self.alpha)
    def create_model_fit(self):
        self._model = Ridge(self.alpha)
        self._model.fit(self.x_train, self.y_train) # Fitting the data into the model
    def tune (self, ds, params):
        self.separator()
        folds = KFold(n_splits=10, shuffle=True, random_state=self.seed)
        #params = {'alpha':[0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1, 5,10,20]}
        model = Ridge()
        model_cv = GridSearchCV(estimator=model, param_grid=params, scoring='r2', cv=folds, return_train_score=True)
        model_cv.fit(self.x_train, self.y_train)
        self.best_params = model_cv.best_params_
        return self.best_params

