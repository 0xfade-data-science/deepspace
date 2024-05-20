import sys
from IPython.display import display, HTML

class Base():
    use_sep = True
    verbose = True
    def __init__(self, sep='=', nb=30, verbose=True):
        self.sep = sep
        self.nb = nb
        self.verbose = verbose
    def __str__(self):
        return type(self).__name__
    def print(self, *args):
        #if Base.verbose:
        print(*args)

    def display(self, df):
        display(HTML(df.to_html()))

    def separator(self, n=1, string='', nb=None, caller='', sep=None):
        if Base.use_sep :
            sep = sep if sep else self.sep
            nb = nb if nb else self.nb
            print(sep * nb, caller, sys._getframe(n).f_code.co_name, string, "\n", end="")

    def minus_one(self, e, arr):
        a = []
        for x in arr:
            if x != e:
                a.append(x)
        return a
    def minus_many(self, arr, to_remove_from):
        a = []
        for x in arr:
            if x not in to_remove_from:
                a.append(x)
        return a
    def intercept(self, arr1, arr2):
        a = []
        for x in arr1:
            if x in arr2:
                a.append(x)
        return a
    #used in inverse separators
    def _get_ordered_cols(self, columns1, columns2):
        ordered_cols = []
        for col in columns1:
            if col in columns2:
                ordered_cols.append(col)
        return ordered_cols        
