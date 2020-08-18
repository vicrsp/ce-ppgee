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


def plot_execution_summary(ga_results, ga_results_2):
    mean_fitness = [ga_results.descale(np.mean(v))
                    for v in ga_results.generation_fitness]
    best_fitness = [ga_results.descale(np.max(v))
                    for v in ga_results.generation_fitness]

    mean_fitness2 = [ga_results.descale(np.mean(v))
                     for v in ga_results_2.generation_fitness]
    best_fitness2 = [ga_results.descale(np.max(v))
                     for v in ga_results_2.generation_fitness]

    _, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(mean_fitness[:40], 'b')
    ax[0].plot(best_fitness[:40], 'k')
    ax[0].legend(['Média', 'Melhor'])
    ax[0].set_xlabel('Geração')

    ax[0].set_ylabel('Função objetivo')

    ax[1].plot(mean_fitness2[:40], 'b')
    ax[1].plot(best_fitness2[:40], 'k')
    ax[1].legend(['Média', 'Melhor'])
    ax[1].set_xlabel('Geração')
    ax[1].set_ylabel('Função objetivo')

    ax[0].set_title('População = 10')
    ax[1].set_title('População = 100')


def pop_size_results(n=10):
    m = Rastrigin(n)

    ga_instance_pop10 = GA([-5.12]*n, [5.12]*n, m.f,
                           pop_size=10, num_bits=20)

    ga_instance_pop100 = GA([-5.12]*n, [5.12]*n, m.f,
                            pop_size=100, num_bits=20)

    ga_instance_pop10.run()
    ga_instance_pop100.run()

    ga_instance_pop10.save_results('pop10')
    ga_instance_pop100.save_results('pop100')

    plot_execution_summary(ga_instance_pop10, ga_instance_pop100)

    plt.tight_layout()
    plt.show()


def generate_rastrigin_statistics(pop_size, runs=30, n=10):
    m = Rastrigin(n)

    final_best_objective = []
    final_best_sol = []
    test_best_fitness = []
    test_mean_fitness = []
    final_mean_fitness = []

    for i in range(runs):
        ga_instance = GA([-5.12]*n, [5.12]*n, m.f,
                         pop_size=pop_size, num_bits=20)
        ga_instance.run()

        ga_instance.save_results(i)

        mean_fitness = [np.mean(v) for v in ga_instance.generation_fitness]
        best_fitness = [np.max(v) for v in ga_instance.generation_fitness]

        test_best_fitness.append(best_fitness)
        test_mean_fitness.append(mean_fitness)

        final_best_objective.append(ga_instance.best_objective)
        final_best_sol.append(ga_instance.best_solution)
        final_mean_fitness.append(ga_instance.descale(
            np.mean(ga_instance.population_fitness)))

        print('BEST SOL: {}'.format(ga_instance.best_solution))
        print('BEST FOBJ: {}'.format(ga_instance.best_objective))
        print('=================================================')

    bs_best_fitness = bs.bootstrap(
        np.array(final_best_objective), stat_func=bs_stats.mean)

    bs_mean_fitness = bs.bootstrap(
        np.array(final_mean_fitness), stat_func=bs_stats.mean)

    # print(statistics.describe())
    print('Melhor solução final: {} CI 95% ({}, {})'.format(bs_best_fitness.value,
                                                            bs_best_fitness.lower_bound, bs_best_fitness.upper_bound))

    print('Melhor solução MÉDIA final: {} CI 95% ({}, {})'.format(bs_mean_fitness.value,
                                                                  bs_mean_fitness.lower_bound, bs_mean_fitness.upper_bound))

    return test_best_fitness, test_mean_fitness, bs_best_fitness, bs_mean_fitness, final_best_sol


if __name__ == "__main__":
    #results = generate_rastrigin_statistics(runs=30, n=10, pop_size=50)
    pop_size_results()
