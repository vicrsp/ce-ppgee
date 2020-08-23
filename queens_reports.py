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


def plot_scenarios(results_1, results_2, title1, title2, title3='Distribuição de geração de convergência'):

    fig, ax = plt.subplots()

    max_gen_1 = results_1.groupby(['idx']).max()['generation']
    max_gen_2 = results_2.groupby(['idx']).max()['generation']

    sns.distplot(max_gen_1, bins=10, kde=False, ax=ax)
    sns.distplot(max_gen_2, bins=10, kde=False, ax=ax)
    ax.legend([title1, title2])
    ax.set_xlabel('Geração')
    ax.set_ylabel('Quantidade')
    ax.set_title(title3)

    return fig


def pop_size_results():
    data_pop100 = run_scenario(pop_size=100, id='pop_10')
    data_pop10 = run_scenario(pop_size=10, id='pop_100')

    data_pop10.to_csv('data/queens_scenario_pop10.csv')
    data_pop100.to_csv('data/queens_scenario_pop100.csv')

    fig = plot_scenarios(data_pop10, data_pop100,
                         'População = 10', 'População = 100')

    plt.tight_layout()
    fig.savefig('queens_scenario_pop_variation.png')


def run_scenario(n=8, runs=30, pop_size=100, crossover_probability=0.6, mutation_probability=0.1, use_inversion_mutation=True, id=''):
    m = NQueens(n)

    data = pd.DataFrame(
        columns=['idx', 'generation', 'type', 'value'])

    converge_count = 0

    for i in range(runs):
        ga_instance = GAPermutation(m.f, pop_size=pop_size, num_generations=300, mutation_probability=mutation_probability,
                                    crossover_probability=crossover_probability, use_inversion_mutation=use_inversion_mutation)
        ga_instance.run()

        mean_fitness = [ga_instance.descale(np.mean(v))
                        for v in ga_instance.generation_fitness]
        best_fitness = [ga_instance.descale(np.max(v))
                        for v in ga_instance.generation_fitness]
        data_instance = transform_data(i, ga_instance,
                                       mean_fitness, best_fitness)

        ga_instance.save_results('queens_scenario_{}_{}'.format(id, i))

        data = data.append(data_instance, ignore_index=True)

        if(ga_instance.converged):
            converge_count = converge_count + 1

    print('Convergence rate: {}'.format(converge_count/float(runs)))
    return data


def crossover_results():

    data_xmin = run_scenario(pop_size=100, crossover_probability=0.3)
    data_xmax = run_scenario(pop_size=100, crossover_probability=0.9)

    data_xmin.to_csv('data/queens_scenario_crossoverMIN.csv')
    data_xmax.to_csv('data/queens_scenario_crossoverMAX.csv')

    fig = plot_scenarios(data_xmin, data_xmax,
                         'Crossover: 0.3', 'Crossover: 0.9')

    plt.tight_layout()
    fig.savefig('queens_scenario_crossover_variation.png')


def mutation_results():

    data_xmin = run_scenario(
        pop_size=100, mutation_probability=0.01)
    data_xmax = run_scenario(
        pop_size=100, mutation_probability=0.2)

    data_xmin.to_csv('data/queens_scenario_mutationMIN.csv')
    data_xmax.to_csv('data/queens_scenario_mutationMAX.csv')

    fig = plot_scenarios(data_xmin, data_xmax,
                         'Mutação: 0.001', 'Mutação: 0.2')

    plt.tight_layout()
    fig.savefig('queens_scenario_mutation_variation.png')


def mutation_results_inversion():

    data_xmin = run_scenario(
        pop_size=100)
    data_xmax = run_scenario(
        pop_size=100, use_inversion_mutation=True)

    data_xmin.to_csv('data/queens_scenario_mutationSwap.csv')
    data_xmax.to_csv('data/queens_scenario_mutationInversion.csv')

    fig = plot_scenarios(data_xmin, data_xmax,
                         'Swap Mutation', 'Inversion Mutation')

    plt.tight_layout()
    fig.savefig('queens_scenario_mutation_inversion.png')


def transform_data(i, ga_instance_xovermin, mean_fitness_xmin, best_fitness_xmin):
    data_min = pd.DataFrame(columns=['idx', 'generation', 'type', 'value'])
    n = len(mean_fitness_xmin)
    data_min['idx'] = [i]*(n*2)
    data_min['generation'] = np.array([np.arange(n), np.arange(n)]).flatten()
    data_min['type'] = np.array([['Média'] * n, ['Melhor'] * n]).flatten()
    data_min['value'] = np.array(
        [mean_fitness_xmin, best_fitness_xmin]).flatten()
    return data_min


if __name__ == "__main__":
    # pop_size_results()
    # crossover_results()
    # mutation_results()
    mutation_results_inversion()

    # datamin = pd.read_csv('data/queens_scenario_mutationMIN.csv')
    # datamax = pd.read_csv('data/queens_scenario_mutationMAX.csv')

    # sns.lineplot(x="generation", y="value",
    #              data=datamin[datamin['type'] == 'Média'])
    # sns.lineplot(x="generation", y="value",
    #              data=datamax[datamax['type'] == 'Média'])
    # plt.legend(['Mutação: 0.001', 'Mutação: 0.2'])
    # plt.tight_layout()
    # plt.xlabel('Geração')
    # plt.ylabel('Função objetivo')
    # plt.savefig('queens_scenario_mutation_variation_average.png')
