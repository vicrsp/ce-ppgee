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


def plot_scenarios(results_1, results_2, title1, title2):

    fig, ax = plt.subplots(2, 1, sharex=True)
    sns.lineplot(data=results_1[results_1['generation'] < 50], x='generation',
                 y='value', hue='type', ax=ax[0])
    ax[0].legend(['Média', 'Melhor'])
    ax[0].set_xlabel('Geração')
    ax[0].set_ylabel('Função objetivo')

    sns.lineplot(data=results_2[results_2['generation'] < 50], x='generation',
                 y='value', hue='type', ax=ax[1])
    ax[1].legend(['Média', 'Melhor'])
    ax[1].set_xlabel('Geração')
    ax[1].set_ylabel('Função objetivo')

    ax[0].set_title(title1)
    ax[1].set_title(title2)

    return fig


def plot_execution_summary_crossover(results_1, results_2):

    _, ax = plt.subplots(2, 1, sharex=True)
    sns.lineplot(data=results_1, x='generation',
                 y='value', hue='type', ax=ax[0])
    ax[0].legend(['Média', 'Melhor'])
    ax[0].set_xlabel('Geração')
    ax[0].set_ylabel('Função objetivo')

    sns.lineplot(data=results_2, x='generation',
                 y='value', hue='type', ax=ax[1])
    ax[1].legend(['Média', 'Melhor'])
    ax[1].set_xlabel('Geração')
    ax[1].set_ylabel('Função objetivo')

    ax[0].set_title('Crossover: [0.6-0.8]')
    ax[1].set_title('Crossover: [0.8-1.0]')


def pop_size_results():

    #data_pop200 = pd.read_csv('scenario_pop_200.csv')
    # data_pop100 = run_scenario(pop_size=200, id='pop_10')
    data_pop10 = run_scenario(pop_size=10, id='pop_100')
    data_pop200 = run_scenario(pop_size=200, id='pop_200')

    # data_pop10 = pd.read_csv('data/scenario_pop10.csv')
    # data_pop100.to_csv('scenario_pop100.csv')

    data_pop10.to_csv('data/scenario_pop_10.csv')
    data_pop200.to_csv('data/scenario_pop_200.csv')

    fig = plot_scenarios(data_pop10, data_pop200,
                         'População = 10', 'População = 200')

    plt.tight_layout()
    fig.savefig('scenario_pop_variation_v2.png')


def bits_results():
    data_bits10 = run_scenario(pop_size=100, num_bits=10, id='bits_10')
    data_bits30 = run_scenario(pop_size=100, num_bits=30, id='bits_30')

    data_bits10.to_csv('scenario_bits10.csv')
    data_bits30.to_csv('scenario_bits30.csv')

    fig = plot_scenarios(data_bits10, data_bits30,
                         'L = 10', 'L = 30')

    plt.tight_layout()
    fig.savefig('scenario_bits_variation.png')


def linear_scaling_results():
    data_ls = run_scenario(
        pop_size=100, linear_scaling=True,  id='linear_scaling')
    data_nolc = run_scenario(
        pop_size=100, linear_scaling=False, id='no_linear_scaling')

    data_ls.to_csv('scenario_linear_scaling.csv')
    data_nolc.to_csv('scenario_no_linear_scaling.csv')

    fig = plot_scenarios(data_ls, data_nolc,
                         'Com escala linear', 'Sem escala linear')

    plt.tight_layout()
    fig.savefig('scenario_linear_scaling.png')


def run_scenario(n=10, runs=30, pop_size=100, num_bits=20, min_crossover_probability=0.6, max_crossover_probability=0.9, min_mutation_probability=0.01, max_mutation_probability=0.05, linear_scaling=False, id=''):
    m = Rastrigin(n)

    data = pd.DataFrame(
        columns=['idx', 'generation', 'type', 'value'])

    for i in range(runs):
        ga_instance = GA([-5.12]*n, [5.12]*n, m.f,
                         pop_size=pop_size, num_bits=num_bits, min_crossover_probability=min_crossover_probability, max_crossover_probability=max_crossover_probability, linear_scaling=linear_scaling)
        ga_instance.run()

        mean_fitness = [np.nanmean(v)
                        for v in ga_instance.generation_fobj]
        best_fitness = [np.nanmin(v)
                        for v in ga_instance.generation_fobj]
        data_instance = transform_data(i, ga_instance,
                                       mean_fitness, best_fitness)

        ga_instance.save_results('scenarion_{}_{}'.format(id, i))

        data = data.append(data_instance, ignore_index=True)

    return data


def crossover_results():

    data_xmin = run_scenario(
        pop_size=100, min_crossover_probability=0.6, max_crossover_probability=0.8, id='crossover_min')
    data_xmax = run_scenario(
        pop_size=100, min_crossover_probability=0.8, max_crossover_probability=1.0, id='crossover_max')

    data_xmin.to_csv('scenario_crossoverMIN.csv')
    data_xmax.to_csv('scenario_crossoverMAX.csv')

    fig = plot_scenarios(data_xmin, data_xmax,
                         'Crossover: [0.6-0.8]', 'Crossover: [0.8-1.0]')

    plt.tight_layout()
    fig.savefig('scenario_crossover_variation.png')


def mutation_results():

    data_xmin = run_scenario(
        pop_size=100, min_mutation_probability=0.01, max_mutation_probability=0.1, id='mutation_min')
    data_xmax = run_scenario(
        pop_size=100, min_mutation_probability=0.1, max_mutation_probability=0.2, id='mutation_max')

    data_xmin.to_csv('scenario_mutationMIN.csv')
    data_xmax.to_csv('scenario_mutationMAX.csv')

    fig = plot_scenarios(data_xmin, data_xmax,
                         'Mutação: [0.01-0.1]', 'Mutação: [0.1-0.2]')

    plt.tight_layout()
    fig.savefig('scenario_mutation_variation.png')


def transform_data(i, ga_instance_xovermin, mean_fitness_xmin, best_fitness_xmin):
    data_min = pd.DataFrame(columns=['idx', 'generation', 'type', 'value'])
    data_min['idx'] = [i]*(ga_instance_xovermin.num_generations*2)
    data_min['generation'] = np.array([np.arange(ga_instance_xovermin.num_generations), np.arange(
        ga_instance_xovermin.num_generations)]).flatten()
    data_min['type'] = np.array([['Média'] * ga_instance_xovermin.num_generations, [
        'Melhor'] * ga_instance_xovermin.num_generations]).flatten()
    data_min['value'] = np.array(
        [mean_fitness_xmin, best_fitness_xmin]).flatten()
    return data_min


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
    # results = generate_rastrigin_statistics(runs=30, n=10, pop_size=50)
    pop_size_results()
    # crossover_results()
    # mutation_results()
    # bits_results()
    # linear_scaling_results()

    # data_10 = pd.read_csv('data/scenario_bits10.csv')
    # data_30 = pd.read_csv('data/scenario_bits30.csv')

    # data_10.groupby('generation').mean()
    # data_30.groupby('generation').mean()

    # fig = plot_scenarios(data_ls, data_nolc,
    #                      'Com escala linear', 'Sem escala linear')

    # plt.tight_layout()
    # fig.savefig('scenario_linear_scaling.png')
