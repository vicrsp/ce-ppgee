from ga_rastrigin import GA
from rastrigin import Rastrigin


def victor_ruela(nvar, ncal):
    m = Rastrigin(nvar)
    ga_instance = GA([-5.12]*nvar, [5.12]*nvar, m.f, max_feval=ncal)
    ga_instance.run()

    return ga_instance.best_solution, ga_instance.best_objective


if __name__ == "__main__":
    x, f = victor_ruela(10, 10000)
