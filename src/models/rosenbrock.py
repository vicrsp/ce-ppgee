from scipy.optimize import rosen


class Rosenbrock:
    def __init__(self, n=10):
        self.n = n

    def f(self, x, index):
        return rosen(x)
