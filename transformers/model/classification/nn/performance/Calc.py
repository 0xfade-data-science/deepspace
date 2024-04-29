import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import pandas as pd
from deepspace.DataSpace import DataSpace
from deepspace.base import Base

from deepspace.transformers.model.classification.logistic.sklearn.LogisticRegression.performance.Calc import Calc as PerformanceCalculator

class Calc(PerformanceCalculator):
    def __init__(self, threshold=0.5) : 
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self, threshold=threshold)

    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()
        self.show_loss()
        self.show_threshold_tuning()
        return ds
    
    def _get_bin_prediction_from_threshold(self, predicted):
        binary_pred = predicted > self.threshold
        return binary_pred
    
    def get_prediction(self, model, predictors, kind='default'):
        return self._get_bin_prediction(model, predictors)

    def get_perf__X(self, x, y):
        self.separator(caller=self)
        yp = self.get_bin_prediction(self._model, x)
        self.train_scores_df = self.as_dataframe(y, yp)
        self.display(self.train_scores_df)
        return yp, self.train_scores_df
    
    def show_loss(self):
        # Capturing learning history per 
        self.history = self.ds.history
        hist  = pd.DataFrame(self.history.history)
        #hist['epoch'] = self.history.epoch
        self.display(hist.head())

        # Plotting accuracy at different epochs
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.legend(("train" , "valid") , loc =0)
        plt.show()        

        #Printing results
        results = self._model.evaluate(self.ds.x_test, self.ds.y_test)
        self.print(results)

    def show_threshold_tuning(self):
        # predict probabilities
        yhat1 = self._get_prediction(self._model, self.ds.x_test)
        # keep probabilities for the positive outcome only
        yhat1 = yhat1[:, 0]
        # calculate roc curves
        fpr, tpr, thresholds1 = roc_curve(self.ds.y_test, yhat1)
        # calculate the g-mean for each threshold
        gmeans1 = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans1)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds1[ix], gmeans1[ix]))
        # plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        plt.show()        

        self.threshold = thresholds1[ix]
        self.ds.perf_train_tuned, self.ds.perf_test_tuned = self.calc_perf()
