from scipy.optimize import rosen
from numpy import exp


class Rosenbrock:
    def __init__(self, n=10):
        self.n = n

    def f(self, x, index):
        y = rosen(x)
        return exp(-y)
