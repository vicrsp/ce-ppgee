import numpy as np


class NQueens:
    def __init__(self, n=8):
        self.n = n

    def f(self, x):
        f = 0
        max_colissions = self.n*(self.n-1)/2
        for i in range(self.n):
            for j in range(self.n):
                if(np.abs(i-j) == np.abs(x[i] - x[j])) & (i != j):
                    f = f + 1
        return f / 2
