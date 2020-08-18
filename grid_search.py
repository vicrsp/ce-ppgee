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


def generate_rastrigin_statistics(pop_size, mutation_probability, crossover_probability, runs=30, n=10):
    m = Rastrigin(n)

    final_best_fitness = []
    final_best_sol = []
    test_best_fitness = []
    test_mean_fitness = []

    for _ in range(runs):
        ga_instance = GA([-5.12]*n, [5.12]*n, m.f,
                         num_generations=10000, mutation_probability=mutation_probability, pop_size=pop_size, crossover_probability=crossover_probability)
        ga_instance.run()

        mean_fitness = [np.mean(v) for v in ga_instance.generation_fitness]
        best_fitness = [np.max(v) for v in ga_instance.generation_fitness]

        test_best_fitness.append(best_fitness)
        test_mean_fitness.append(mean_fitness)

        final_best_fitness.append(ga_instance.best_objective)
        final_best_sol.append(ga_instance.best_objective)

    # Generate statistics table
    statistics = pd.DataFrame()
    statistics['Melhor solução final'] = np.array(final_best_fitness)

    bs_best_fitness = bs.bootstrap(
        np.array(final_best_fitness), stat_func=bs_stats.mean)

    # print(statistics.describe())
    print('Melhor solução final: {} CI 95% ({}, {})'.format(bs_best_fitness.value,
                                                            bs_best_fitness.lower_bound, bs_best_fitness.upper_bound))

    return test_best_fitness, test_mean_fitness, bs_best_fitness, final_best_sol, statistics


if __name__ == "__main__":

    # Mutation Results
    bs_best_fitness_results = pd.DataFrame(
        np.zeros((27, 6)), columns=['pm', 'pc', 'pop', 'fobj', 'ci_lb', 'ci_ub'])
    bs_mean_fitness_results = bs_best_fitness_results.copy()

    mut_cross_results = {}
    mutations = [0.01, 0.05, 0.1]
    crossovers = [0.7, 0.8, 0.9]
    pop_sizes = [10, 20, 50]
    for i in range(3):
        for k in range(3):
            for j in range(3):
                mut_cross_results['{}_{}_{}'.format(i, k, j)] = generate_rastrigin_statistics(
                    runs=30, n=10, pop_size=pop_sizes[j], mutation_probability=mutations[i], crossover_probability=crossovers[k])

    index = 0
    for key in mut_cross_results:
        test_best_fitness, test_mean_fitness, bs_mean_fitness, bs_best_fitness, statistics = mut_cross_results[
            key]

        pd.DataFrame(test_best_fitness).T.to_csv(
            'best_fitness_{}.csv'.format(key), sep=";")

        pd.DataFrame(test_mean_fitness).T.to_csv(
            'mean_fitness_{}.csv'.format(key), sep=";")

        m, c, pop = key.split('_')

        bs_best_fitness_results.loc[index, :] = [mutations[int(m)], crossovers[int(c)], pop_sizes[int(
            pop)], bs_best_fitness.value, bs_best_fitness.lower_bound, bs_best_fitness.upper_bound]
        bs_mean_fitness_results.loc[index, :] = [mutations[int(m)], crossovers[int(c)], pop_sizes[int(
            pop)], bs_mean_fitness.value, bs_mean_fitness.lower_bound, bs_mean_fitness.upper_bound]

        statistics.to_csv('statistics_{}.csv'.format(key), sep=";")

        index = index + 1

    bs_best_fitness_results.to_csv(
        'bs_best_fitness_results_mutcross.csv', sep=";")
    bs_mean_fitness_results.to_csv(
        'bs_mean_fitness_results_mutcross.csv', sep=";")
