import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from src.models.rosenbrock import Rosenbrock
from src.models.rastrigin import Rastrigin
from src.models.test_functions import Quadratic
from src.models.queens import NQueens

from src.optimization.ga import GA
from src.optimization.ga_permutation import GAPermutation

from math import log
from mpl_toolkits import mplot3d


def nqueens_plot(ga_results):
    mean_fitness = [28 - np.mean(v) for v in ga_results.generation_fitness]
    best_fitness = [28 - np.max(v) for v in ga_results.generation_fitness]

    _, ax = plt.subplots()
    ax.plot(mean_fitness, 'b.')
    ax.plot(best_fitness, 'k.')
    ax.legend(['Mean fitness', 'Best fitness'])
    ax.set_xlabel('Generation #')
    ax.set_ylabel('Fitness')

    # best_solutions = np.empty(
    #     (ga_results.num_generations, 8))
    # for generation in ga_results.generation_solutions.keys():
    #     fitness = ga_results.generation_fitness[generation]
    #     ibest = np.argmax(fitness)
    #     sbest = ga_results.generation_solutions[generation][ibest, :]
    #     best_solutions[generation, :] = sbest

    # _, ax1 = plt.subplots(8, 1)

    # for i in range(8):
    #     ax1[i].plot(best_solutions[:, i])
    #     ax1[i].set_ylabel('x_{}'.format(i))

    image = np.zeros((8, 8))

    for index, val in enumerate(ga_results.best_solution):
        image[index, int(val)] = 1

    plt.matshow(image)
    plt.xticks(range(8), range(8))
    plt.yticks(range(8), range(8))

    plt.show()


def data_for_contour_plot(lb, ub, fitness):
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
    mean_fitness = [np.nanmean(v) for v in ga_results.generation_fobj]
    best_fitness = [np.nanmin(v) for v in ga_results.generation_fobj]

    fig, ax = plt.subplots()
    ax.plot(mean_fitness, 'b.')
    ax.plot(best_fitness, 'k.')
    ax.legend(['Mean fitness', 'Best fitness'])
    ax.set_xlabel('Generation #')
    ax.set_ylabel('Fitness')

    _, ax1 = plt.subplots(ga_results.num_variables, 1)

    best_solutions = np.empty(
        (ga_results.num_generations, ga_results.num_variables))
    for generation in ga_results.generation_solutions.keys():
        fitness = ga_results.generation_fitness[generation]
        ibest = np.argmax(fitness)
        sbest = ga_results.generation_solutions[generation][ibest, :]
        best_solutions[generation, :] = sbest

    for i in range(ga_results.num_variables):
        ax1[i].plot(best_solutions[:, i])
        ax1[i].set_ylabel('x_{}'.format(i))

    # plot the contour for dim=2
    if(ga_results.num_variables == 2):
        fig, ax = plt.subplots()
        x, y, z = data_for_contour_plot(ga_results.lb, ga_results.ub,
                                        ga_results.fitness_func)

        cs = ax.contourf(x, y, z, cmap="RdBu_r", levels=15)
        fig.colorbar(cs, ax=ax)
        # ax.plot(best_solutions[:, 0], best_solutions[:, 1], 'ko--', ms=5)
    plt.show()


def run_queens():
    """Executa o algoritmo GA para a função das N rainhas
    """
    m = NQueens()
    ga_instance = GAPermutation(m.f,
                                num_generations=500, mutation_probability=0.1, pop_size=100, crossover_probability=0.8, use_inversion_mutation=True)
    ga_instance.run()
    nqueens_plot(ga_instance)


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
                     num_generations=1000, pop_size=100, linear_scaling=True)
    ga_instance.run()

    # ga_instance.save_results('teste')

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
    # run_rastrigin()
    # run_quadratic()
    # run_rosenbrock()
    run_queens()
