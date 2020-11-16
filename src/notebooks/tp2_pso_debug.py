# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from timeit import default_timer as timer
from mpl_toolkits import mplot3d
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% [markdown]
# # Computação Evolucionária - TP PSO
# %% [markdown]
# ## Introdução
# %% [markdown]
# ## Metodologia
#
# ### Função unimodal
# A função unimodal escolhida foi a 5 (Different Powers Function)
#
# ### Função multimodal
# A função multimodal escolhida foi a 10 (Rotated Griewank's Function)
# %% [markdown]
# ### Desenvolvimento das funções de avaliação
#
# Inicialmente, é necessário importar as bibliotecas necessárias para o desenvolvimento do trabalho. Está sendo utilizada a versão 3.8.5 do Python, e as bibliotecas `numpy`, `pandas`, `matplotlib`, `mpl_toolkits` e `seaborn`.

# %%

# importando bibliotecas

# %% [markdown]
# O proximo passo é implementar e validar a implementação das funções de teste escolhidas para estudo.

# %%


class DifferentPowersFunction:
    """
    Função 5 - Different Powers Function
    """

    def __init__(self, o, D=10, f_star=-1000):
        self.o = o
        self.D = D
        self.f_star = f_star

    def func(self, x):
        z = x - self.o
        z_sum = np.sum([(np.abs(z[i])) ** (2 + 4*(i)/(self.D-1))
                        for i in np.arange(start=0, stop=self.D)])
        return np.sqrt(np.sum(z_sum)) + self.f_star


# %%
class RotatedGriewanksFunction:
    """
    Função 10 - Rotated Griewanks's Function
    """

    def __init__(self, o, M1, alpha=100, D=10, f_star=-500):
        self.o = o
        self.D = D
        self.f_star = f_star
        self.M1 = M1
        self.diag = np.diag([alpha ** (i/(2*(D-1)))
                             for i in np.arange(self.D)])

    def calculate_z(self, x):
        return self.diag @ self.M1 @ (6 * (x - self.o))

    def func(self, x):
        z = self.calculate_z(x)
        part1 = np.sum([(z[i]**2)/4000 for i in np.arange(self.D)])
        part2 = -np.prod([np.cos(z[i]/np.sqrt(i+1))
                          for i in np.arange(self.D)])
        return part1 + part2 + 1 + self.f_star


# %%
def plot_function_2d(func, n, bx=[-100, 100], by=[-100, 100], title=''):
    """
    Plota a superfície e curvas de níveis para uma função de testes
    """

    lb, ub = bx
    x = np.linspace(lb, ub, n)
    lb, ub = by
    y = np.linspace(lb, ub, n)
    xv, yv = np.meshgrid(x, y)
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z[i, j] = func(np.array([xv[i, j], yv[i, j]]))

    fig = plt.figure(num=None, figsize=(12, 6))
    fig.suptitle(title)

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.plot_surface(xv, yv, z, cmap='jet', edgecolor='none')
    cs = ax2.contour(x, y, z, cmap='jet', levels=50)
    fig.colorbar(cs, ax=ax2)
    plt.show()


# %%
# Ler os dados de shift fornecidos
shift_data = pd.read_fwf('src/notebooks/data/shift_data.txt', header=None)
# Ler a matriz para duas dimensões
M_D2 = pd.read_fwf('src/notebooks/data/M_D2.txt', header=None)
# número de dimensões para gráficos
D_plot = 2
# selecionar um único valor de shift
o_used = shift_data.iloc[0, :D_plot].to_numpy()
# selecionar duas matrizes de rotação
M1 = M_D2.iloc[:2, :].to_numpy()

unimodal_2d = DifferentPowersFunction(o_used, D=D_plot)
multimodal_2d = RotatedGriewanksFunction(o_used, M1, D=D_plot)

# plot_function_2d(unimodal_2d.func, 100,
#                  title='Função 5 (Different Powers Function)')
# plot_function_2d(multimodal_2d.func, 100,
#                  bx=[-26, -17], by=[7, 16], title='Função 10 (Rotated Griewank’s Function)')

# %% [markdown]
# Através dos gráficos acima, é possível ver que as funções implementadas retornam os mesmos valores exibidos em [1]. Logo, o próximo passo consiste em implementar o algoritmo PSO e executar os experimentos propostos para o trabalho.
#
# ### Desenvolvimento do algoritmo PSO
#
# O algoritmo PSO foi desenvolvido na classe `PSO`, definda na célula a seguir. A implementação será validada sobre uma função quadrática simples em duas dimensões para avaliar o seu correto funcionamento. É esperada uma convergência rápida para o mínimo local em $x^* = (0,0)$.
#

# %%


class PSO:
    def __init__(self, func, lb, ub, max_feval=10000, swarm_size=100, acceleration=[0.01, 0.01], constrition=1, inertia=0.7, topology='gbest'):
        self.max_feval = max_feval
        self.func = func
        self.swarm_size = swarm_size
        self.num_variables = len(lb)
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.topology = topology
        self.x = constrition
        self.w = inertia
        self.c1, self.c2 = acceleration
        self.window = 10

    def reset(self):
        self.fevals = 0
        self.global_best_cost = np.Infinity
        self.personal_best_cost = np.ones(self.swarm_size) * np.Infinity
        self.personal_best_solution = np.zeros(
            (self.swarm_size, self.num_variables))
        self.iteration_average_cost = []
        self.iteration_global_best_cost = []

    def initialize_swarm(self):
        """
        Initializes the swarm
        """
        self.swarm = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.swarm_size, self.num_variables))
        self.swarm_velocity = np.zeros_like(self.swarm)

    def evaluate_swarm_cost_function(self, swarm):
        """
        Calculating the cost values of all solutions in the current swarm.
        """
        swarm_cost = []
        for index, sol in enumerate(swarm):
            cost = self.func(sol)
            swarm_cost.append(cost)
            # applies barrier function
            # if(np.any(sol < self.lb) | np.any(sol > self.ub)):
            #     cost = np.Infinity
            #     self.swarm_velocity[index, :] = np.zeros_like(sol)

            # Update global best cost
            if(cost < self.global_best_cost):
                self.global_best_cost = cost
                self.global_best_solution = sol

            # Update the personal best position
            if(cost < self.personal_best_cost[index]):
                self.personal_best_cost[index] = cost
                self.personal_best_solution[index, :] = sol

        swarm_cost = np.array(swarm_cost)

        self.fevals = self.fevals + swarm_cost.shape[0]
        self.swarm_cost = swarm_cost

        # Store iteration data
        self.iteration_average_cost.append(np.mean(swarm_cost))
        self.iteration_global_best_cost.append(self.global_best_cost)

        return swarm_cost

    def get_social_component_solution(self, swarm, swarm_cost):
        if(self.topology == 'gbest'):
            return np.array([swarm[np.argmin(swarm_cost), :]] * self.swarm_size)
        elif(self.topology == 'lbest'):
            solution = np.zeros_like(swarm)
            for i in range(self.swarm_size):
                neighbors = np.vstack(
                    (swarm[(i-1) % self.swarm_size, :], swarm[i, :], swarm[(i+1) % self.swarm_size, :]))
                neighbors_cost = np.array([swarm_cost[(
                    i-1) % self.swarm_size], swarm_cost[i], swarm_cost[(i+1) % self.swarm_size]])
                best_neighbor = np.argmin(neighbors_cost)
                solution[i, :] = neighbors[best_neighbor, :]

            return solution

    def update_particles_velocity(self, swarm, swarm_cost):
        updated_velocities = []
        updated_swarm = []

        social_best_sol = self.get_social_component_solution(
            self.personal_best_solution, self.personal_best_cost)

        for i, particle in enumerate(swarm):
            r1 = np.random.rand(self.num_variables)
            r2 = np.random.rand(self.num_variables)

            cognitive_component = self.c1 * r1 * \
                (self.personal_best_solution[i, :] - particle)
            social_component = self.c2 * r2 * \
                (social_best_sol[i, :] - particle)
            velocity = self.w * \
                self.swarm_velocity[i, :] + \
                cognitive_component + social_component
            new_velocity = self.x * \
                np.sign(velocity) * np.minimum(np.abs(velocity),
                                               self.ub.max() * np.ones_like(velocity))

            updated_velocities.append(new_velocity)
            updated_swarm.append(particle + new_velocity)

        self.swarm_velocity = np.array(updated_velocities)
        self.swarm = np.array(updated_swarm)

    def validate_boundaries(self, swarm):
        for particle in swarm:
            reflection_diff_max = self.lb - particle
            reflection_diff_min = particle - self.ub

            zeros_part = np.zeros_like(particle)
            particle = particle + \
                2 * np.maximum(zeros_part, reflection_diff_min) + \
                2 * np.maximum(zeros_part, reflection_diff_max)
            particle = np.minimum(self.ub, particle)

    def run(self, f_star=None, debug=True):
        # reset progress variables
        self.reset()

        start = timer()
        self.initialize_swarm()
        num_generations = 0
        while(self.fevals < self.max_feval):
            # Calculating the cost of each particle in the swarm.
            self.evaluate_swarm_cost_function(self.swarm)
            self.update_particles_velocity(self.swarm, self.swarm_cost)
            # self.validate_boundaries(self.swarm)

            num_generations = num_generations + 1

            if(num_generations > self.window):
                # print(np.mean(np.abs(np.diff(self.iteration_average_cost[-self.window:]))))
                curr_tol = np.mean(
                    np.abs(np.diff(self.iteration_average_cost[-self.window:])))
                if (curr_tol < 0.0001):
                    print('stop due to y_tol - {}'.format(num_generations))
                    break

        end = timer()
        print('Elapsed: {}'.format(end - start))

        if(debug):
            self.report()

        return self.global_best_solution, self.global_best_cost, self.iteration_average_cost, self.iteration_global_best_cost

    def report(self):
        print('Best particle: {}'.format(self.global_best_solution))
        print('Best cost: {}'.format(self.global_best_cost))
        self.plot_charts()

    def plot_charts(self):
        _, ax = plt.subplots()
        ax.plot(self.iteration_average_cost, 'b.-')
        ax.plot(self.iteration_global_best_cost, 'r.-')
        plt.legend(['Swarm average cost', 'Global best cost'])
        plt.tight_layout()
        plt.show()


# %%
# criar instancia da função de teste quadrática
def quad_func(x): return (x[0] - 1)**2 + (x[1] - 2)**2


# Executar o algoritmo
pso_unimodal = PSO(quad_func, [-10, -10], [10, 10], max_feval=10000,
                   swarm_size=50, acceleration=[2, 2], constrition=1.0, inertia=1.0)
_, _, _, _ = pso_unimodal.run()

pso_unimodal = PSO(quad_func, [-10, -10], [10, 10], max_feval=10000, swarm_size=30,
                   acceleration=[1.5, 1.5], constrition=1.0, inertia=1.0, topology='lbest')
_, _, _, _ = pso_unimodal.run()

# %% [markdown]
# A partir dos resultados anteriores, é possível constatar que o algoritmo consegue convergir com facilidade para uma funçao simples de teste para as estruturas de vizinhança `gbest` e `lbest (anel)`. Dessa forma, é possível considerar sua implementação como validada.
# %% [markdown]
# ## Resultados
# Inicialmente é necessário importar os dados necessários e definir as variáveis para as instâncias do problema a serem estudadas.

# %%
# Ler a matriz M
M_D10 = pd.read_fwf('src/notebooks/data/M_D10.txt', header=None)

# Definições padrões para os testes
D = 10
n_executions = 31
max_fevals = 100000
ub = [100]*D
lb = [-100]*D

# selecionar um único valor de shift
o_test = shift_data.iloc[0, :D].to_numpy()
# selecionar matriz de rotação
M1 = M_D10.iloc[:D, :].to_numpy()

# %% [markdown]
# Em seguida, basta instanciar um objeto único que representa cada função de avaliação com os dados importados.

# %%
# Função unimodal
unimodal = DifferentPowersFunction(o_test, D=D)
# Funcão multimodal
multimodal = RotatedGriewanksFunction(o_test, M1, D=D)

# %% [markdown]
# Como forma de entender como cada hiper-parâmetro existente afeta o desempenho do algoritmo PSO para cada uma das estruturas de vizinhança, a seguir serão avaliados os parâmetros tamanho da população e coeficientes de aceleração. Por meio da análise dos resultados, serão definidos os valores que consigam um bom desempenho para as duas estruturas de vizinhança, os quais serão posteriormente utilizados na análise proposta para os parâmetros de constrição e peso de inércia. Será considerado como critério de parada 10000 avaliações da função objetivo.
#
# Inicialmente, é definida uma função de apoito responsável por executar de forma fácil um experimento variando os hiperparâmetros disponíveis, e també para gerar os relatórios que sejam relevantes.
#

# %%


def run_experiment(func, constrition=0.5, inertia=0.5, n_runs=5, max_feval=10000, swarm_size=30, acceleration=[1.5, 1.5], f_star=None):
    data = np.zeros((n_runs, 2))
    data_progress_gbest = []
    data_progress_lbest = []
    pso_lbest = PSO(func, lb, ub, max_feval, swarm_size,
                    acceleration, constrition, inertia, topology='lbest')
    pso_gbest = PSO(func, lb, ub, max_feval, swarm_size,
                    acceleration, constrition, inertia, topology='gbest')

    for i in range(n_runs):
        best_solution_lbest, best_cost_lbest, global_best_lbest, _ = pso_lbest.run(
            debug=False, f_star=f_star)
        best_solution_gbest, best_cost_gbest, global_best_gbest, _ = pso_gbest.run(
            debug=False, f_star=f_star)

        data_progress_gbest.append(np.array(global_best_gbest))
        data_progress_lbest.append(np.array(global_best_lbest))

        data[i, 0] = best_cost_lbest
        data[i, 1] = best_cost_gbest

    results = pd.DataFrame(data, columns=['lbest', 'gbest'])
    results['constrition'] = [constrition] * n_runs
    results['inertia'] = [inertia] * n_runs
    results['swarm_size'] = [swarm_size] * n_runs
    results['acceleration_1'] = [acceleration[0]] * n_runs
    results['acceleration_2'] = [acceleration[1]] * n_runs

    return results, pd.DataFrame(data_progress_gbest).T, pd.DataFrame(data_progress_lbest).T


def plot_experiment_results(gbest_results, lbest_results):
    n_results = len(gbest_results)
    fig, ax = plt.subplots(n_results, 2, sharey=True,
                           sharex=True, figsize=(12, 8))

    for i in range(n_results):
        gbest = gbest_results[i]
        lbest = lbest_results[i]

        mean_gbest = gbest.apply(np.mean, axis=1)
        mean_lbest = lbest.apply(np.mean, axis=1)
        x_gbest = range(1, mean_gbest.shape[0] + 1)
        x_lbest = range(1, mean_lbest.shape[0] + 1)

        std_gbest = gbest.apply(np.std, axis=1)
        std_lbest = lbest.apply(np.std, axis=1)

        ax[i, 0].plot(x_gbest, mean_gbest)
        ax[i, 0].fill_between(x_gbest, mean_gbest + std_gbest,
                              mean_gbest - std_gbest, alpha=0.5)

        ax[i, 1].plot(x_lbest, mean_lbest, 'r')
        ax[i, 1].fill_between(x_lbest, mean_lbest + std_lbest,
                              mean_lbest - std_lbest, alpha=0.5, color='red')

        if(i == 0):
            ax[i, 0].set_title('gbest')
            ax[i, 1].set_title('lbest')

    plt.show()


def generate_experiment_statistics(experiments, group_name):
    all_data = pd.concat(experiments, ignore_index=True)
    stats = all_data.groupby(group_name)[
        ['lbest', 'gbest']].aggregate([np.mean, np.std])
    print(stats)

# %% [markdown]
# ### Tamanho da população
#
# Para este experimento, serão consideradas populações de tamanho 10, 30, 50 e 100. Os demais parâmetros serão mantidos fixos em seus valores padrão, conforme definidos na assinatura da função `run_experiment`.


# %%
# execute the experiment
swarm_sizes = [10, 30, 50, 100]
results_multimodal = []
results_unimodal = []
for swarm_size in swarm_sizes:
    results_unimodal.append(run_experiment(
        unimodal.func, swarm_size=swarm_size))
    results_multimodal.append(run_experiment(
        multimodal.func, swarm_size=swarm_size))
    print('finish')


# %%
# unimodal function statistics
generate_experiment_statistics([value[0]
                                for value in results_unimodal], ['swarm_size'])
#plot_experiment_results([value[1] for value in results_unimodal], [value[2] for value in results_unimodal])

# %% [markdown]
# Para a função unimodal, é possível notar que é possível chegar mais próximo do mínimo global da função com a versão `gbest`. O melhor resultado é obtido com a maior população testada, com 100 indivíduos. Isso sugere que é necessário um melhor ajuste dos demais hiper-pâmetros para se obter melhores resultados em ambas as versões para esta função.

# %%
# multimodal function statistics
generate_experiment_statistics(
    [value[0] for value in results_multimodal], ['swarm_size'])
plot_experiment_results([value[1] for value in results_multimodal], [
                        value[2] for value in results_multimodal])

# %% [markdown]
# Já para a função multimodal, a versão `lbest` apresenta melhores resultados, sendo importante notar que o aumento do tamanho da população acima de 50 indivíduos não aparenta introduzir melhoras para o seu desempenho. No caso da versão `gbest`, populações maiores que 50 melhorar um pouco o desempenho do algoritmo, embora os resultados ainda sejam piores.
#
# Tendo em vista os resultados apresentados, uma população de 50 indivíduos será considerada para os demais experimentos.
# %% [markdown]
#
# ### Coeficientes de aceleração
#

# %%
# execute the experiment
accel_coef = [1.0, 1.33, 1.66, 2.0]
results_multimodal_accel = []
results_unimodal_accel = []
for coef in accel_coef:
    results_unimodal_accel.append(run_experiment(
        unimodal.func, swarm_size=50, acceleration=[coef, coef]))
    results_multimodal_accel.append(run_experiment(
        multimodal.func, swarm_size=50, acceleration=[coef, coef]))
    print('finish')


# %%
# unimodal function statistics
generate_experiment_statistics([value[0] for value in results_unimodal_accel], [
                               'acceleration_1', 'acceleration_2'])


# %%
# multimodal function statistics
generate_experiment_statistics([value[0] for value in results_multimodal_accel], [
                               'acceleration_1', 'acceleration_2'])

# %% [markdown]
# ### Parâmetro de constrição
#

# %%
# execute the experiment
constritions = [0.0, 0.33, 0.66, 1.0]
results_multimodal_constrition = []
results_unimodal_constrition = []
for constrition in constritions:
    results_unimodal_constrition.append(run_experiment(
        unimodal.func, swarm_size=50, acceleration=[2.0, 2.0], constrition=constrition))
    results_multimodal_constrition.append(run_experiment(
        multimodal.func, swarm_size=50, acceleration=[1.66, 1.66], constrition=constrition))
    print('finish')


# %%
# unimodal function statistics
generate_experiment_statistics(
    [value[0] for value in results_unimodal_constrition], ['constrition'])

# %% [markdown]
#

# %%
# multimodal function statistics
generate_experiment_statistics(
    [value[0] for value in results_multimodal_constrition], ['constrition'])

# %% [markdown]
# ### Peso de inércia

# %%
# execute the experiment
inertias = [0.0, 0.33, 0.66, 1.0]
results_multimodal_inertia = []
results_unimodal_inertia = []
for inertia in inertias:
    results_unimodal_inertia.append(run_experiment(unimodal.func, swarm_size=50, acceleration=[
                                    2.0, 2.0], constrition=1.0, inertia=inertia))
    results_multimodal_inertia.append(run_experiment(multimodal.func, swarm_size=50, acceleration=[
                                      1.66, 1.66], constrition=1.0, inertia=inertia))
    print('finish')


# %%
# unimodal function statistics
generate_experiment_statistics(
    [value[0] for value in results_unimodal_inertia], ['inertia'])


# %%
# multimodal function statistics
generate_experiment_statistics(
    [value[0] for value in results_multimodal_inertia], ['inertia'])

# %% [markdown]
# ## Resultados
# ## Função unimodal

# %%

pss = PSO(multimodal.func, lb, ub, max_feval=10000, swarm_size=30,
          acceleration=[2.0, 2.0], constrition=1, inertia=1, topology='gbest')
_, _, _, _ = pss.run(debug=True)


# %%
# execute the experiment
parameters = [{'inertia': 1, 'constrition': 1}, {'inertia': 0.3, 'constrition': 1}, {
    'inertia': 1, 'constrition': 0.3}, {'inertia': 0.6, 'constrition': 0.6}]
results_unimodal = []
for params in parameters:
    print(params)
    results_unimodal.append(run_experiment(unimodal.func, f_star=unimodal.f_star, swarm_size=50, acceleration=[
                            1.1, 1.1], max_feval=max_fevals, constrition=params['constrition'], inertia=params['inertia']))


# %%
generate_experiment_statistics([value[0] for value in results_unimodal], [
                               'constrition', 'inertia'])

# %% [markdown]
# ## Função multimodal

# %%
# execute the experiment
parameters = [{'inertia': 1, 'constrition': 1}, {'inertia': 0.3, 'constrition': 1}, {
    'inertia': 1, 'constrition': 0.3}, {'inertia': 0.6, 'constrition': 0.6}]
results_multimodal = []
for params in parameters:
    print(params)
    results_multimodal.append(run_experiment(multimodal.func, f_star=multimodal.f_star, swarm_size=50, acceleration=[
                              2.0, 2.0], max_feval=max_fevals, constrition=params['constrition'], inertia=params['inertia']))


# %%
generate_experiment_statistics([value[0] for value in results_multimodal], [
                               'constrition', 'inertia'])

# %% [markdown]
# ## Conclusão
# %% [markdown]
# ## Referências
#
# [[1]] J. J. Liang, B-Y. Qu, P. N. Suganthan, Alfredo G. Hern´andez-D´ıaz, "Problem Definitions and Evaluation
# Criteria for the CEC 2013 Special Session and Competition on Real-Parameter Optimization", Technical Report 201212, Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou China and
# Technical Report, Nanyang Technological University, Singapore, January 2013.

# %%
