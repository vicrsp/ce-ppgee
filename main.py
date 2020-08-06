import numpy as np
import matplotlib.pyplot as plt
from src.models.rosenbrock import Rosenbrock
from src.optimization.ga import GA


def run_queens():
    """Executa o algoritmo GA para a função das N rainhas
    """


def run_rosenbrock():
    """
    Executa o algoritmo GA para a função de rosenbrock
    """
    m = Rosenbrock()
    ga_instance = GA([-5, -5], [10, 10], m.f,
                     num_generations=100, mutation_probability=1)
    ga_instance.run()
    print(ga_instance)


if __name__ == "__main__":
    run_rosenbrock()
