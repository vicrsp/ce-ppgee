import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


data_best = pd.read_csv('best_fitness_1_2_2.csv', sep=';')
data_mean = pd.read_csv('mean_fitness_1_2_2.csv', sep=';')

mean_melt = data_mean.melt(id_vars='Unnamed: 0', value_vars=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'])

best_melt = data_best.melt(id_vars='Unnamed: 0', value_vars=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'])
mean_melt['Resultado'] = ['Médio'] * mean_melt.shape[0]
best_melt['Resultado'] = ['Melhor'] * best_melt.shape[0]

mean_melt_filt = mean_melt.loc[mean_melt['Unnamed: 0'].astype('int') < 70]
best_melt_filt = best_melt.loc[best_melt['Unnamed: 0'].astype('int') < 70]

all_vars = mean_melt_filt.append(best_melt_filt, ignore_index=True)

# data_mean['ci']
#sns.lineplot(data=mean_melt, x='Unnamed: 0', y='value')
#sns.lineplot(data=best_melt, x='Unnamed: 0', y='value')
sns.lineplot(data=all_vars, x='Unnamed: 0', y='value', hue='Resultado')
plt.ylabel('Função objetivo')
plt.xlabel('# Geração')
plt.tight_layout()
plt.show()
# sns.lineplot(x="timepoint", y="signal", hue="region",
#              style="event", data=data_mean)
