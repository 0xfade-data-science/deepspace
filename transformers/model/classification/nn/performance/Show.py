import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer

class ShowNOTUSED(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.ds = ds
        self._model = ds._model
        self.show_perf()
        return ds
    def show_perf(self):
        # Capturing learning history per epoch
        self.history = self._model.history
        hist  = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch

        # Plotting accuracy at different epochs
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.legend(("train" , "valid") , loc =0)

        #Printing results
        results = self._model.evaluate(self.ds.x_test, self.ds.y_test)

        #self.show_confusion_matrix()
    def make_confusion_matrix(self):
        #Calculating the confusion matrix
        cm=confusion_matrix(y_test, y_pred)
        labels = ['True Negative','False Positive','False Negative','True Positive']
        categories = [ 'Not Changing Job','Changing Job']
        make_confusion_matrix(cm,
                            group_names=labels,
                            categories=categories,
                            cmap='Blues')        
    def show_confusion_matrix(cf,
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


        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names)==cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            #Accuracy is sum of diagonal divided by total observations
            accuracy  = np.trace(cf) / float(np.sum(cf))



        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False


        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)


        if title:
            plt.title(title)