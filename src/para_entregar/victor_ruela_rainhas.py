from ga_rainhas import GAPermutation
from queens import NQueens


def victor_ruela(nvar):
    m = NQueens(nvar)
    ga_instance = GAPermutation(m.f, max_int=nvar)
    ga_instance.run()
    return ga_instance.best_solution, ga_instance.best_objective


if __name__ == "__main__":
    x, f = victor_ruela(10)
