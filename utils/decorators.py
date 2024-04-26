########################################################################
######### Decorators #########################################
########################################################################

def printcall(sep='x', n=50):
    def decorator(func):
        #@functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-decoration logic using arg1 and arg2
            separator = sep*n
            print(f'{separator} called {func.__qualname__}')
            result = func(*args, **kwargs)
            # Post-decoration logic
            return result
        return wrapper
    return decorator

@printcall(sep='_')
def my_function(phrase):
    # Function code
    print(phrase)
class X:
    @printcall()
    def truc(self, x):
        print(x**2)