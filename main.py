import numpy as np
import matplotlib.pyplot as plt
from src.models.rosenbrock import Rosenbrock
from src.models.rastrigin import Rastrigin
from src.models.test_functions import Quadratic
from src.optimization.ga import GA
from math import log
from mpl_toolkits import mplot3d


def data_for_contour_plot(lb, ub, fitness,):
    n = 100
    x = np.linspace(lb[0], ub[0], n)
    y = np.linspace(lb[1], ub[1], n)
    z = np.empty((n, n))

    for i in range(n):
        for j in range(n):
            z[i, j] = log(
                1 / fitness(np.array([x[i], y[j]]), 0))

    return x, y, z


def plot_execution_summary(ga_results):
    mean_fitness = [log(1 / np.mean(v)) for v in ga_results.generation_fitness]
    best_fitness = [log(1 / np.max(v)) for v in ga_results.generation_fitness]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(mean_fitness, 'b.')
    ax[0].plot(best_fitness, 'k.')
    ax[0].legend(['Mean fitness', 'Best fitness'])
    ax[0].set_xlabel('Generation #')
    ax[0].set_ylabel('Fitness')

    # plot the contour for dim=2
    if(ga_results.num_variables == 2):
        x, y, z = data_for_contour_plot(ga_results.lb, ga_results.ub,
                                        ga_results.fitness_func)

        cs = ax[1].contourf(x, y, z, cmap="RdBu_r", levels=15)
        fig.colorbar(cs, ax=ax[1])
        best_solutions = np.empty((ga_results.num_generations, 2))
        for generation in ga_results.generation_solutions.keys():
            fitness = ga_results.generation_fitness[generation]
            ibest = np.argmax(fitness)
            sbest = ga_results.generation_solutions[generation][ibest, :]
            best_solutions[generation, :] = sbest

        ax[1].plot(best_solutions[:, 0], best_solutions[:, 1], 'ko--', ms=5)
    plt.show()


def run_queens():
    """Executa o algoritmo GA para a função das N rainhas
    """


def run_quadratic():
    """
    Executa o algoritmo GA para a função quadratica
    """
    m = Quadratic()
    ga_instance = GA([-10]*2, [10]*2, m.f,
                     num_generations=100, mutation_probability=0.001, pop_size=10)
    ga_instance.run()

    plot_execution_summary(ga_instance)


def run_rastrigin(n=10):
    """
    Executa o algoritmo GA para a função de rastrigin
    """
    m = Rastrigin(n)
    ga_instance = GA([-5.12]*n, [5.12]*n, m.f,
                     num_generations=50, mutation_probability=0.1, pop_size=20, crossover_probability=0.7)
    ga_instance.run()

    plot_execution_summary(ga_instance)


def run_rosenbrock():
    """
    Executa o algoritmo GA para a função de rosenbrock
    """
    m = Rosenbrock()
    ga_instance = GA([-1]*2, [1]*2, m.f,
                     num_generations=50, mutation_probability=0.1, crossover_probability=0.7, pop_size=10)
    ga_instance.run()

    plot_execution_summary(ga_instance)


if __name__ == "__main__":
    run_rastrigin(2)
    # run_quadratic()
    # run_rosenbrock()
