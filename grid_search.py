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


def generate_rastrigin_statistics(num_generations, pop_size, mutation_probability, crossover_probability,  runs=30, n=10):
    m = Rastrigin(n)
    # _, ax_vars = plt.subplots(n, 1)
    # _, ax_fitness = plt.subplots(2, 1)

    total_generations = []
    total_feval = []
    final_best_fitness = []
    final_mean_fitness = []
    final_best_sol = []
    final_mean_sol = []
    test_best_fitness = []
    test_mean_fitness = []

    for _ in range(runs):
        ga_instance = GA([-5.12]*n, [5.12]*n, m.f,
                         num_generations=num_generations, mutation_probability=mutation_probability, pop_size=pop_size, crossover_probability=crossover_probability)
        ga_instance.run()

        mean_fitness = [log(1 / np.mean(v))
                        for v in ga_instance.generation_fitness]
        best_fitness = [log(1 / np.max(v))
                        for v in ga_instance.generation_fitness]

        test_best_fitness.append(best_fitness)
        test_mean_fitness.append(mean_fitness)

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

        # ax_fitness[0].plot(mean_fitness)
        # ax_fitness[1].plot(best_fitness)

        # best_solutions = np.empty(
        #     (ga_instance.num_generations, ga_instance.num_variables))
        # for generation in ga_instance.generation_solutions.keys():
        #     fitness = ga_instance.generation_fitness[generation]
        #     ibest = np.argmax(fitness)
        #     sbest = ga_instance.generation_solutions[generation][ibest, :]
        #     best_solutions[generation, :] = sbest

        # for i in range(ga_instance.num_variables):
        #     ax_vars[i].plot(best_solutions[:, i])
        #     ax_vars[i].set_ylabel('x_{}'.format(i))

    # Generate statistics table
    statistics = pd.DataFrame()
    statistics['Total de Gerações'] = np.array(total_generations)
    statistics['Total de Avaliações de Fitness'] = np.array(total_feval)
    statistics['Média das soluções finais'] = np.array(final_mean_fitness)
    statistics['Melhor solução final'] = np.array(final_best_fitness)

    bs_generations = bs.bootstrap(
        np.array(total_generations), stat_func=bs_stats.mean)
    bs_feval = bs.bootstrap(np.array(total_feval),
                            stat_func=bs_stats.mean)
    bs_mean_fitness = bs.bootstrap(
        np.array(final_mean_fitness), stat_func=bs_stats.mean)
    bs_best_fitness = bs.bootstrap(
        np.array(final_best_fitness), stat_func=bs_stats.mean)

    # print(statistics.describe())
    print('Total de Gerações: {} CI 95% ({}, {})'.format(int(bs_generations.value),
                                                         int(bs_generations.lower_bound), int(bs_generations.upper_bound)))
    print('Total de Avaliações de Fitness: {} CI 95% ({}, {})'.format(int(bs_feval.value),
                                                                      int(bs_feval.lower_bound), int(bs_feval.upper_bound)))
    print('Média das soluções finais: {} CI 95% ({}, {})'.format(bs_mean_fitness.value,
                                                                 bs_mean_fitness.lower_bound, bs_mean_fitness.upper_bound))

    print('Melhor solução final: {} CI 95% ({}, {})'.format(bs_best_fitness.value,
                                                            bs_best_fitness.lower_bound, bs_best_fitness.upper_bound))
    # # Plot densities
    # _, ax1 = plt.subplots(2, 2)
    # sns.distplot(statistics['Total de Gerações'], ax=ax1[0, 0])
    # sns.distplot(statistics['Total de Avaliações de Fitness'], ax=ax1[0, 1])
    # sns.distplot(statistics['Média das soluções finais'], ax=ax1[1, 0])
    # sns.distplot(statistics['Melhor solução final'], ax=ax1[1, 1])

    return test_best_fitness, test_mean_fitness, bs_feval, bs_mean_fitness, bs_best_fitness, statistics


if __name__ == "__main__":

    # pop_generation_results = {}
    # pop_sizes = [10, 20, 50]
    # gen_sizes = [1000, 500, 200]

    # # Pop vs Gen Results size result
    # for i in range(3):
    #     pop_generation_results['pop_gen_{}'.format(i)] = generate_rastrigin_statistics(
    #         runs=30, n=10, num_generations=gen_sizes[i], pop_size=pop_sizes[i], mutation_probability=0.05, crossover_probability=0.8)

    # _, ax_fitness = plt.subplots(3, 2)
    # bs_feval_results = pd.DataFrame(np.zeros((3, 3)), columns=[
    #                                 'value', 'ci_lb', 'ci_ub'])
    # bs_best_fitness_results = bs_feval_results.copy()
    # bs_mean_fitness_results = bs_feval_results.copy()

    # index = 0
    # for key in pop_generation_results:
    #     test_best_fitness, test_mean_fitness, bs_feval, bs_mean_fitness, bs_best_fitness, statistics = pop_generation_results[
    #         key]

    #     pd.DataFrame(test_best_fitness).T.to_csv(
    #         'best_fitness_{}.csv'.format(key), sep=";")

    #     pd.DataFrame(test_mean_fitness).T.to_csv(
    #         'mean_fitness_{}.csv'.format(key), sep=";")

    #     bs_feval_results.loc[index, :] = [bs_feval.value,
    #                                       bs_feval.lower_bound, bs_feval.upper_bound]
    #     bs_best_fitness_results.loc[index, :] = [
    #         bs_best_fitness.value, bs_best_fitness.lower_bound, bs_best_fitness.upper_bound]
    #     bs_mean_fitness_results.loc[index, :] = [
    #         bs_mean_fitness.value, bs_mean_fitness.lower_bound, bs_mean_fitness.upper_bound]

    #     statistics.to_csv('statistics_{}.csv'.format(key), sep=";")

    #     index = index + 1

    # bs_feval_results.to_csv('popgen_bs_feval_results.csv', sep=";")
    # bs_best_fitness_results.to_csv(
    #     'popgen_bs_best_fitness_results_popgen.csv', sep=";")
    # bs_mean_fitness_results.to_csv(
    #     'popgen_bs_mean_fitness_results_popgen.csv', sep=";")

    # Mutation Results
    bs_feval_results = pd.DataFrame(np.zeros((9, 3)), columns=[
                                    'value', 'ci_lb', 'ci_ub'])
    bs_best_fitness_results = bs_feval_results.copy()
    bs_mean_fitness_results = bs_feval_results.copy()

    mut_cross_results = {}
    mutations = [0.01, 0.05, 0.1]
    crossovers = [0.7, 0.8, 0.9]
    for i in range(3):
        for k in range(3):
            mut_cross_results['mut_{},cross_{}'.format(i, k)] = generate_rastrigin_statistics(
                runs=30, n=10, num_generations=1000, pop_size=10, mutation_probability=mutations[i], crossover_probability=crossovers[k])

    index = 0
    for key in mut_cross_results:
        test_best_fitness, test_mean_fitness, bs_feval, bs_mean_fitness, bs_best_fitness, statistics = mut_cross_results[
            key]

        pd.DataFrame(test_best_fitness).T.to_csv(
            'best_fitness_{}.csv'.format(key), sep=";")

        pd.DataFrame(test_mean_fitness).T.to_csv(
            'mean_fitness_{}.csv'.format(key), sep=";")

        bs_feval_results.loc[index, :] = [bs_feval.value,
                                          bs_feval.lower_bound, bs_feval.upper_bound]
        bs_best_fitness_results.loc[index, :] = [
            bs_best_fitness.value, bs_best_fitness.lower_bound, bs_best_fitness.upper_bound]
        bs_mean_fitness_results.loc[index, :] = [
            bs_mean_fitness.value, bs_mean_fitness.lower_bound, bs_mean_fitness.upper_bound]

        statistics.to_csv('statistics_{}.csv'.format(key), sep=";")

        index = index + 1

    bs_feval_results.to_csv('bs_feval_results_mutcross.csv', sep=";")
    bs_best_fitness_results.to_csv(
        'bs_best_fitness_results_mutcross.csv', sep=";")
    bs_mean_fitness_results.to_csv(
        'bs_mean_fitness_results_mutcross.csv', sep=";")
