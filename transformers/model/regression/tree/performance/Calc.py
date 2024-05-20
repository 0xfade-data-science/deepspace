import pandas as pd


from deepspace.base import Base
from deepspace.DataSpace import DataSpace
from deepspace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator

class Calc(PerformanceCalculator):
    def __init__(self) : 
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator .__init__(self)
    def transform_remove(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()
        return ds
    def from_ds_init_remove(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds._model
    def performance(self, predictors, target):
        #pdb.set_trace()
        model = self.get_model()
        print(type(model))
        prediction = model.predict(predictors) # Predict
        #pred = np.reshape(pred, (pred.shape[0], 1))
        prediction = pd.DataFrame(prediction)
        # dataframe of metrics to make it easy to view and work with
        df_perf = self.as_dataframe(predictors, prediction, target)
        return df_perf, prediction

