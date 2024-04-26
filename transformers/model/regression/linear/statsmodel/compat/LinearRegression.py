#make OLS Model compatible with cross_val_score
# from https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
class MyOLS_As_LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    """
    Parameters
    ------------
    column_names: list
            It is an optional value, such that this class knows
            what is the name of the feature to associate to
            each column of X. This is useful if you use the method
            summary(), so that it can show the feature name for each
            coefficient
    """
    def fit(self, X, y, column_names=() ):
        if self.fit_intercept:
            X = sm.add_constant(X)
        # Check that X and y have correct shape
        #TODO but it returns ndarray and not pands Series
        #X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        if len(column_names) != 0:
            cols = column_names.copy()
            cols = list(cols)
            X = pd.DataFrame(X)
            cols = column_names.copy()
            cols.insert(0,'intercept')
            print('X ', X)
            X.columns = cols
        self.model_ = sm.OLS(y, X)
        self.results_ = self.model_.fit()
        #print('type = ', type(self.results_.params))
        return self
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')
        # Input validation
        X = check_array(X)
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)
    def get_params(self, deep = False):
        return {'fit_intercept':self.fit_intercept}
    def summary(self):
        print(self.results_.summary() )
class Tr_MyOLSModelTransformerCompat(MyOLS_As_LinearRegression):
    def __init__(self, fit_intercept=False) :
        MyBaseClass.__init__(self, '=', 50)
        #Tr_MyBaseModelTransformer.__init__(self) #commented because fit takes 1 arg only
        MyOLS_As_LinearRegression.__init__(self,  fit_intercept=fit_intercept) # we need 3 args
        self._model = self.fitted_model = None
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self.x_train = sm.add_constant(self.x_train)
        self.x_test = sm.add_constant(self.x_test)
        ds.x_train = self.x_train
        ds.x_test = self.x_test
        MyOLS_As_LinearRegression.fit(self, self.x_train, self.y_train)
        self._model = self.model_
        # Fitting the model
        ds.fitted_model = self.fitted_model = self.results_
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
    def init_ds(self, ds):
        ds._model = self.get_model()
    def get_model(self):
        return self._model
    def create_model_fit(self):
        pass
    def create_model_fit(self):
        pass
    def tune(self):
        pass

