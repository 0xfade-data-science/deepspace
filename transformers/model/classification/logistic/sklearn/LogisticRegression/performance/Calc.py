import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, accuracy_score, f1_score, precision_recall_curve

from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base
#from DeepSpace.transformers.outliers.Check import CheckOutliers

from DeepSpace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator

class Calc(PerformanceCalculator):
    def __init__(self, threshold=0.5):
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self)
        self.threshold = threshold
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()
        self.precision_recall_curve()
        self.show_perf_with_given_threshold()
        return ds
    def from_ds_init(self, ds):
        self.ds = ds
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds.get_model()
    def get_model(self):
        return self._model
    def calc_perf(self):
        self.separator(caller=self)
        return self.perf_train(), self.perf_test()
    def get_perf(self, x, y):
        self.separator(caller=self)
        yp = self.get_prediction(self._model, x)
        scores_df = self.show_metrics_score(y, yp)
        self.plot_confusion_matrix(y, yp)
        return yp, scores_df
    def perf_train(self):
        self.separator(caller=self)
        self.y_train_pred, self.train_scores_df = self.get_perf(self.x_train, self.y_train)
        return self.train_scores_df
    def perf_test(self):
        self.separator(caller=self)
        self.y_test_pred, self.test_scores_df = self.get_perf(self.x_test, self.y_test)
        return self.test_scores_df
    
    def _get_bin_prediction_from_threshold(self, predicted):
        binary_pred = predicted < self.threshold
        return binary_pred
    def _get_bin_prediction_proba_from_threshold(self, predicted):
        binary_pred = predicted[:, 1] < self.threshold # we have 2 classes 0 and 1, need class 1
        return binary_pred
    def _get_bin_prediction(self, model, predictors):
        pred = self._get_prediction(model, predictors)
        return self._get_bin_prediction_from_threshold(pred)
    def _get_prediction_proba(self, model, predictors):
        pred = model.predict_proba(predictors)
        return pred
    def _get_prediction(self, model, predictors):
        return model.predict(predictors)
    def get_prediction(self, model, predictors, kind='default'):
        if kind == 'default':
            return self._get_prediction(model, predictors)
        if kind == 'binary':
            return self._get_bin_prediction(model, predictors)
        if kind == 'proba':
            return self._get_prediction_proba(model, predictors)
    def show_metrics_score(self, actual, predicted):
        self.separator(caller=self)
        print(classification_report(actual, predicted))
        self.calc_scores(actual, predicted)
        scores_df = self.get_scores_as_dataframe()
        #scores_df = self.as_dataframe(actual, predicted)
        self.display(scores_df)
        return scores_df
    def plot_confusion_matrix(self, actual, predicted, detailed=True):
        target_col = self.ds.target_col
        cm = confusion_matrix(actual, predicted)
        if not detailed:
            self.plot_confusion_matrix_simple(cm, target_col)
        else:
            self.plot_confusion_matrix_detailed(cm, target_col)
    def plot_confusion_matrix_simple(self, cm, target_col):
        categories=[f'!{target_col}', target_col]
        self._plot_confusion_matrix_simple(cm, categories=categories)
    def _plot_confusion_matrix_simple(cm, categories='auto'):
        plt.figure(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt='.2f', 
                    xticklabels=categories, 
                    yticklabels=categories)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def plot_confusion_matrix_detailed(self, cm, target_col):
        target_col = self.ds.target_col
        labels = ['True Negative','False Positive','False Negative','True Positive']
        categories = [ f'!{target_col}', target_col ]
        self._plot_confusion_matrix_detailed(cm, group_names=labels, categories=categories, cmap='Blues')
    def _plot_confusion_matrix_detailed(
                            self,
                            cm,
                            group_names=None,
                            categories='auto',
                            count=True,
                            percent=True,
                            cbar=True,
                            xyticks=True,
                            xyplotlabels=True,
                            sum_stats=True,
                            figsize=None,
                            cmap='Blues',
                            title=None):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        '''
        #import pdb
        #pdb.set_trace()

        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cm.size)]

        group_labels = group_percentages = group_counts = blanks

        if group_names and len(group_names)==cm.size:
            group_labels = ["{}\n".format(value) for value in group_names]

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cm.shape[0], cm.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cm) / float(np.sum(cm))

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cm,
                    annot=box_labels,
                    fmt="",
                    cmap=cmap,
                    cbar=cbar,
                    xticklabels=categories,
                    yticklabels=categories)

        if title:
            plt.title(title)
        plt.show()

    def calc_scores(self, target, prediction):
        # To compute recall
        self.recall = recall_score(target, prediction, average='macro')
        # To compute precision
        self.precision = precision_score(target, prediction, average='macro')
        # To compute accuracy score
        self.acc = accuracy_score(target, prediction)
        # To compute f1 score
        self.f1 = f1_score(target, prediction)
    def get_scores_as_dataframe(self):
        """
        Function to compute different metrics to check classification model performance
        model: classifier
        predictors: independent variables
        target: dependent variable
        """
        # Creating a dataframe of metrics
        df_perf = pd.DataFrame(
            {
                "Precision":  self.precision,
                "Recall":  self.recall,
                "Accuracy": self.acc,
                "F1": self.f1
            },

            index=[0],
        )
        return df_perf
    def as_dataframe_xxx(self, target, prediction):
        """
        Function to compute different metrics to check classification model performance
        model: classifier
        predictors: independent variables
        target: dependent variable
        """
        # Predicting using the independent variables
        #pred = model.predict(predictors)
        # To compute recall
        recall = recall_score(target, prediction, average='macro')
        # To compute precision
        precision = precision_score(target, prediction, average='macro')
        # To compute accuracy score
        acc = accuracy_score(target, prediction)
        # To compute f1 score
        f1 = f1_score(target, prediction)
        # Creating a dataframe of metrics
        df_perf = pd.DataFrame(
            {
                "Precision":  precision,
                "Recall":  recall,
                "Accuracy": acc,
                "F1": f1,
            },

            index=[0],
        )
        return df_perf

    # only for logistic regressor (no prediction_proba for NN)
    def precision_recall_curve(self):
        self.separator(caller=self)
        y_scores = self._get_prediction_proba(self._model, self.ds.x_train) # predict_proba gives the probability of each observation belonging to each class
        precisions, recalls, thresholds = precision_recall_curve(self.ds.y_train, y_scores[:, 1])
        # Plot values of precisions, recalls, and thresholds
        plt.figure(figsize = (10, 7))
        plt.plot(thresholds, precisions[:-1], 'b--', label = 'precision')
        plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
        plt.xlabel('Threshold')
        plt.legend(loc = 'upper left')
        plt.ylim([0, 1])
        plt.show()

    # only for logistic regressor (no prediction_proba for NN)
    def show_perf_with_given_threshold(self):# only logistic regressor
        self.separator(caller=self, string=f'trying with threshold = {self.threshold}')
        if self.threshold:
            self.print()
            optimal_threshold1 = self.threshold #.35

            y_pred_train = self._get_prediction_proba(self._model, self.ds.x_train)
            self.show_metrics_score(self.ds.y_train, y_pred_train[:, 1] > optimal_threshold1)

            y_pred_test = self._get_prediction_proba(self._model, self.ds.x_test)
            self.show_metrics_score(self.ds.y_test, y_pred_test[:, 1] > optimal_threshold1)

class AbstractShowMetrics(PerformanceCalculator):
    def __init__(self, threshold=0.5):
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self)
        self.threshold = threshold
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.calc(XXX)
        self.display(self.as_dataframe())
        return ds
    def from_ds_init(self, ds):
        self.ds = ds
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds.get_model()
    def get_model(self):
        return self._model
    def calc(self, target, prediction):
        # To compute recall
        self.recall = recall_score(target, prediction, average='macro')
        # To compute precision
        self.precision = precision_score(target, prediction, average='macro')
        # To compute accuracy score
        self.acc = accuracy_score(target, prediction)
        # To compute f1 score
        self.f1 = f1_score(target, prediction)
    def as_dataframe(self, target, prediction):
        """
        Function to compute different metrics to check classification model performance
        model: classifier
        predictors: independent variables
        target: dependent variable
        """
        # Creating a dataframe of metrics
        df_perf = pd.DataFrame(
            {
                "Precision":  self.precision,
                "Recall":  self.recall,
                "Accuracy": self.acc,
                "F1": self.f1
            },

            index=[0],
        )
        return df_perf

