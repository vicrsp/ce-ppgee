import numpy as np
import matplotlib.pyplot as plt
from src.models.rosenbrock import Rosenbrock
from src.models.rastrigin import Rastrigin
from src.models.test_functions import Quadratic
from src.optimization.ga import GA


def run_queens():
    """Executa o algoritmo GA para a função das N rainhas
    """


def run_quadratic():
    """
    Executa o algoritmo GA para a função quadratica
    """
    m = Quadratic()
    ga_instance = GA([-10]*2, [10]*2, m.f,
                     num_generations=100, mutation_probability=0.001, pop_size=30)
    ga_instance.run()

    plt.plot(np.array(ga_instance.best_solutions_fitness))
    plt.show()


def run_rastrigin():
    """
    Executa o algoritmo GA para a função de rastrigin
    """
    m = Rastrigin()
    ga_instance = GA([-5.12]*10, [5.12]*10, m.f,
                     num_generations=200, mutation_probability=0.1, pop_size=20)
    ga_instance.run()

    plt.plot(1 / np.array(ga_instance.best_solutions_fitness))
    plt.show()


def run_rosenbrock():
    """
    Executa o algoritmo GA para a função de rosenbrock
    """
    m = Rosenbrock()
    ga_instance = GA([-5, -5], [10, 10], m.f,
                     num_generations=100, mutation_probability=0.1, crossover_probability=0.7)
    ga_instance.run()

    plt.plot(1 / np.array(ga_instance.best_solutions_fitness))
    plt.show()

    print(ga_instance.best_solutions)


if __name__ == "__main__":
    run_quadratic()
