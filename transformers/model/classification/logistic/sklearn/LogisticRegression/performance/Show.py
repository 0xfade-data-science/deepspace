import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, accuracy_score, f1_score

import pandas as pd

from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base
from DeepSpace.transformers.Transformer import Transformer


class ShowNOTFINISHED(Transformer):
    def __init__(self, threshold=0.5):
        Base.__init__(self, '#!#', 50)
        Transformer.__init__(self)
        self.threshold = threshold
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        pdf = pd.concat([ds.perf_train, ds.perf_test], keys=['train', 'test'], ignore_index=False)
        self.display(pdf)
        self.importance_view(self._model)
        self.metrics_score()
        return ds
    def from_ds_init(self, ds):
        self.ds = ds
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds.get_model()
    def get_model(self):
        return self._model
    def calc_perf(self):
        self.separator()
        return self.perf_train(), self.perf_test()
    def perf_train(self):
        self.separator()
        self.train_scores_df = self.as_dataframe(self._model, self.x_train, self.y_train)
        return self.train_scores_df
    def perf_test(self):
        self.separator()
        self.test_scores_df = self.as_dataframe(self._model, self.x_test, self.y_test)
        return self.test_scores_df
    
    def importance_view(self, model):
        importances = model.feature_importances_
        columns = self.ds.x.columns
        self.importance_df = pd.DataFrame(importances, index=columns, columns=[
                                     'Importance']).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(13, 13))
        sns.barplot(x=self.importance_df.Importance, y=self.importance_df.index)
        return self.importance_df
    def get_pred(self, predicted):
        binary_pred = predicted < self.threshold
        return binary_pred
    def metrics_score(self, actual, predicted):
        target_col = self.ds.target_col
        binary_pred = self.get_pred(predicted)
        print(classification_report(actual, binary_pred))
        cm = confusion_matrix(actual, binary_pred)
        plt.figure(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=[
                    f'!{target_col}', target_col], yticklabels=[f'!{target_col}', target_col])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def as_dataframe(self, model, predictors, target):
        """
        Function to compute different metrics to check classification model performance
        model: classifier
        predictors: independent variables
        target: dependent variable
        """
        # Predicting using the independent variables
        pred = model.predict(predictors)
        binary_pred = self.get_pred(pred)
        # To compute recall
        recall = recall_score(target, binary_pred, average='macro')
        # To compute precision
        precision = precision_score(target, binary_pred, average='macro')
        # To compute accuracy score
        acc = accuracy_score(target, binary_pred)
        # To compute f1 score
        f1 = f1_score(target, binary_pred)
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
