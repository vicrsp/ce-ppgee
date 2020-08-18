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


def pop_size_results():
    data_pop100 = run_scenario(pop_size=100, id='pop_10')
    data_pop10 = run_scenario(pop_size=10, id='pop_100')

    data_pop10.to_csv('data/queens_scenario_pop10.csv')
    data_pop100.to_csv('data/queens_scenario_pop100.csv')

    fig = plot_scenarios(data_pop100, data_pop10,
                         'População = 10', 'População = 100')

    plt.tight_layout()
    fig.savefig('data/queens_scenario_pop_variation.png')


def run_scenario(n=10, runs=2, pop_size=100, id=''):
    m = NQueens(n)

    data = pd.DataFrame(
        columns=['idx', 'generation', 'type', 'value'])

    for i in range(runs):
        ga_instance = GA([-5.12]*n, [5.12]*n, m.f, pop_size=pop_size)
        ga_instance.run()

        mean_fitness = [ga_instance.descale(np.mean(v))
                        for v in ga_instance.generation_fitness]
        best_fitness = [ga_instance.descale(np.max(v))
                        for v in ga_instance.generation_fitness]
        data_instance = transform_data(i, ga_instance,
                                       mean_fitness, best_fitness)

        ga_instance.save_results('data/queens_scenario_{}_{}'.format(id, i))

        data = data.append(data_instance, ignore_index=True)

    return data


def crossover_results():

    data_xmin = run_scenario(
        pop_size=100)
    data_xmax = run_scenario(
        pop_size=100)

    data_xmin.to_csv('data/queens_scenario_crossoverMIN.csv')
    data_xmax.to_csv('data/queens_scenario_crossoverMAX.csv')

    fig = plot_scenarios(data_xmin, data_xmax,
                         'Crossover: [0.6-0.8]', 'Crossover: [0.8-1.0]')

    plt.tight_layout()
    fig.savefig('data/scenario_crossover_variation.png')


def mutation_results():

    data_xmin = run_scenario(
        pop_size=100)
    data_xmax = run_scenario(
        pop_size=100)

    data_xmin.to_csv('data/scenario_mutationMIN.csv')
    data_xmax.to_csv('data/scenario_mutationMAX.csv')

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


if __name__ == "__main__":
    # results = generate_rastrigin_statistics(runs=30, n=10, pop_size=50)
    pop_size_results()
    # crossover_results()
    # mutation_results()
    # bits_results()
