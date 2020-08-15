import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# POP SIZE
# Fitness progress plot
data_best_fitness_0 = pd.read_csv(
    'best_fitness_pop_gen_0.csv', sep=";").iloc[:50, 1:]
data_best_fitness_1 = pd.read_csv(
    'best_fitness_pop_gen_1.csv', sep=";").iloc[:50, 1:]
data_best_fitness_2 = pd.read_csv(
    'best_fitness_pop_gen_2.csv', sep=";").iloc[:50, 1:]

data_mean_fitness_0 = pd.read_csv(
    'mean_fitness_pop_gen_0.csv', sep=";").iloc[:50, 1:]
data_mean_fitness_1 = pd.read_csv(
    'mean_fitness_pop_gen_1.csv', sep=";").iloc[:50, 1:]
data_mean_fitness_2 = pd.read_csv(
    'mean_fitness_pop_gen_2.csv', sep=";").iloc[:50, 1:]

# _, ax = plt.subplots(3, 2, sharex=True)
# ax[0, 0].plot(data_best_fitness_0)
# ax[0, 1].plot(data_mean_fitness_0)
# ax[1, 0].plot(data_best_fitness_1)
# ax[1, 1].plot(data_mean_fitness_1)
# ax[2, 0].plot(data_best_fitness_1)
# ax[2, 1].plot(data_mean_fitness_1)


# Density plots
stats_0 = pd.read_csv('statistics_pop_gen_0.csv',
                      sep=";", skipinitialspace=True)
stats_1 = pd.read_csv('statistics_pop_gen_1.csv',
                      sep=";", skipinitialspace=True)
stats_2 = pd.read_csv('statistics_pop_gen_2.csv',
                      sep=";", skipinitialspace=True)

# # Plot densities
_, ax1 = plt.subplots(3, 2)

sns.distplot(stats_0['Total de Avaliações de Fitness'],
             ax=ax1[0, 0], kde=False, axlabel='')
sns.distplot(stats_0['Melhor solução final'],
             ax=ax1[0, 1], axlabel='', kde=False)

sns.distplot(stats_1['Total de Avaliações de Fitness'],
             ax=ax1[1, 0], color='red', axlabel='', kde=False)
sns.distplot(stats_1['Melhor solução final'] * 10**7,
             ax=ax1[1, 1], color='red', axlabel='', kde=False)

sns.distplot(stats_1['Total de Avaliações de Fitness'],
             ax=ax1[2, 0], color='green', axlabel='', kde=False)
sns.distplot(stats_1['Melhor solução final'] * 10**7,
             ax=ax1[2, 1], color='green', axlabel='', kde=False)

ax1[0, 0].set_title('Total de Avaliações do Fitness')
ax1[0, 1].set_title('Melhor solução (x10^7)')
# Statistics
# popgen_bs_best_fitness_results_popgen = pd.read_csv(
#     'popgen_bs_best_fitness_results_popgen.csv', sep=";")
# popgen_bs_feval_results = pd.read_csv('popgen_bs_feval_results.csv', sep=";")
# popgen_bs_mean_fitness_results_popgen = pd.read_csv(
#     'popgen_bs_mean_fitness_results_popgen.csv', sep=";")

# print(popgen_bs_best_fitness_results_popgen)
# print(popgen_bs_feval_results)
# print(popgen_bs_mean_fitness_results_popgen)

plt.show()
