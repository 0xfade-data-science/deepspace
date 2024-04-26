class LinRegModelTransformerWithPerf(LinearRegression, Tr_MyOLSModelTransformerWithPerf):
    def __init__(self, show_perf=True) : #, perfchecker : MyPerformanceChecker):
        MyBaseClass.__init__(self, '=', 50)
        Tr_MyLinRegModelTransformer.__init__(self)
        Tr_MyOLSModelTransformerWithPerf.__init__(self)
    def transform(self, ds:DataSpace):
        Tr_MyLinRegModelTransformer.transform(self, ds)
        if self.show_perf :
            ds.perf_train, ds.perf_test = self.calc_perf()
        return ds