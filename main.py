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

    best_solutions = np.empty(
        (ga_results.num_generations, 8))
    for generation in ga_results.generation_solutions.keys():
        fitness = ga_results.generation_fitness[generation]
        ibest = np.argmax(fitness)
        sbest = ga_results.generation_solutions[generation][ibest, :]
        best_solutions[generation, :] = sbest

    _, ax1 = plt.subplots(8, 1)

    for i in range(8):
        ax1[i].plot(best_solutions[:, i])
        ax1[i].set_ylabel('x_{}'.format(i))

    image = np.zeros((8, 8))

    best_solution = best_solutions[-1, :]
    for index, val in enumerate(best_solution):
        image[index, int(val)] = 1

    plt.matshow(image)
    plt.xticks(range(8), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
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
    mean_fitness = [log(1 / np.mean(v)) for v in ga_results.generation_fitness]
    best_fitness = [log(1 / np.max(v)) for v in ga_results.generation_fitness]

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
                                num_generations=100, mutation_probability=0.1, pop_size=20, crossover_probability=0.8)
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


def generate_rastrigin_statistics(num_generations, pop_size, runs=30, n=10):
    m = Rastrigin(n)
    # _, ax1 = plt.subplots(n, 1)
    total_generations = []
    total_feval = []
    final_best_fitness = []
    final_mean_fitness = []
    final_best_sol = []
    final_mean_sol = []

    for _ in range(runs):
        ga_instance = GA([-5.12]*n, [5.12]*n, m.f,
                         num_generations=num_generations, mutation_probability=0.05, pop_size=pop_size, crossover_probability=0.8)
        ga_instance.run()

        mean_fitness = [log(1 / np.mean(v))
                        for v in ga_instance.generation_fitness]
        best_fitness = [log(1 / np.max(v))
                        for v in ga_instance.generation_fitness]

        fitness = ga_instance.generation_fitness[-1]
        ibest = np.argmax(fitness)
        sbest = ga_instance.generation_solutions[len(
            ga_instance.generation_fitness)-1][ibest, :]

        total_generations.append(len(ga_instance.generation_fitness))
        total_feval.append(ga_instance.fitness_eval)
        final_mean_fitness.append(mean_fitness[-1])
        final_best_fitness.append(best_fitness[-1])
        final_best_sol.append(sbest)
        final_mean_sol.append(np.mean(
            ga_instance.generation_solutions[len(ga_instance.generation_fitness)-1]))

    # Generate statistics table
    statistics = pd.DataFrame()
    # statistics['Total de Gerações'] = np.array(total_generations)
    # statistics['Total de Avaliações de Fitness'] = np.array(total_feval)
    statistics['Média das soluções finais'] = np.array(final_mean_fitness)
    statistics['Melhor solução final'] = np.array(final_best_fitness)

    # bs_generations = bs.bootstrap(
    #     np.array(total_generations), stat_func=bs_stats.mean)
    # bs_feval = bs.bootstrap(np.array(total_feval),
    #                         stat_func=bs_stats.mean)
    bs_mean_fitness = bs.bootstrap(
        np.array(final_mean_fitness), stat_func=bs_stats.mean)
    bs_best_fitness = bs.bootstrap(
        np.array(final_best_fitness), stat_func=bs_stats.mean)

    print(statistics.describe())
    # print('Total de Gerações: {} CI 95% ({}, {})'.format(bs_generations.value,
    #                                                      bs_generations.lower_bound, bs_generations.upper_bound))
    # print('Total de Avaliações de Fitness: {} CI 95% ({}, {})'.format(bs_feval.value,
    #                                                                   bs_feval.lower_bound, bs_feval.upper_bound))
    print('Média das soluções finais: {} CI 95% ({}, {})'.format(bs_mean_fitness.value,
                                                                 bs_mean_fitness.lower_bound, bs_mean_fitness.upper_bound))

    print('Melhor solução final: {} CI 95% ({}, {})'.format(bs_best_fitness.value,
                                                            bs_best_fitness.lower_bound, bs_best_fitness.upper_bound))
    # Plot densities
    _, ax1 = plt.subplots(1, 2)
    # sns.distplot(statistics['Total de Gerações'], ax=ax1[0, 0])
    # sns.distplot(statistics['Total de Avaliações de Fitness'], ax=ax1[0, 1])
    sns.distplot(statistics['Média das soluções finais'], ax=ax1[0])
    sns.distplot(statistics['Melhor solução final'], ax=ax1[1])

    plt.show()


def run_rastrigin(n=10):
    """
    Executa o algoritmo GA para a função de rastrigin
    """
    m = Rastrigin(n)
    ga_instance = GA([-5.12]*n, [5.12]*n, m.f,
                     num_generations=100, mutation_probability=0.1, pop_size=10, crossover_probability=0.8)
    ga_instance.run()

    plot_execution_summary(ga_instance)
    print(ga_instance.fitness_eval)


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
    # run_queens()

    generate_rastrigin_statistics(
        runs=30, n=10, num_generations=100, pop_size=100)
    generate_rastrigin_statistics(
        runs=30, n=10, num_generations=1000, pop_size=10)
