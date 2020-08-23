import numpy as np
import itertools
import random
import pandas as pd
from math import factorial


class GAChromossome:
    def __init__(self, fitness, value=[], generation=0):
        self.value = value
        self.generation = generation
        self.fitness = fitness

    def setValue(self, value):
        self.value = value

    def setFitness(self, fitness):
        self.fitness = fitness


class GA:
    def __init__(self, lb, ub, fitness_func, pop_size=10, num_generations=10, max_feval=10000, crossover_probability=0.8, min_crossover_probability=0.6, max_crossover_probability=0.8, mutation_probability=0.1, min_mutation_probability=0.01, max_mutation_probability=0.05, num_bits=20, tournament_candidates=10, linear_scaling=False):
        self.population_size = pop_size
        self.num_variables = len(lb)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

        self.min_crossover_probability = min_crossover_probability
        self.max_crossover_probability = max_crossover_probability
        self.min_mutation_probability = min_mutation_probability
        self.max_mutation_probability = max_mutation_probability

        self.linear_scaling = linear_scaling

        self.lb = lb
        self.ub = ub
        self.num_generations = int(max_feval / pop_size)
        self.max_feval = max_feval
        self.fitness_func = fitness_func
        self.num_bits = num_bits  # number of bits to encode each gene
        self.tournament_candidates = tournament_candidates

        self.precision = (np.array(self.ub) - np.array(self.lb)
                          )/(np.exp2(self.num_bits) - 1)
        self.best_solutions_fitness = []
        self.generation_fitness = []
        self.generation_fobj = []
        self.generation_solutions = {}
        self.generation_parents = {}
        self.generation_offspring_mutated = {}
        self.generation_offspring_crossover = {}

        self.K_scaling = 1.5
        self.x_tol = 10 ** -6
        self.fitness_tol = 10 ** -6
        self.fitness_eval = 0
        self.scale_factor = 1
        self.best_objective = np.Infinity
        self.best_solution = []
        self.best_fitness = 0

    def save_results(self, test_id):
        pd.DataFrame(self.best_solution).to_csv(
            'best_solution_{}.csv'.format(test_id), index=False)

        pd.DataFrame([self.best_objective]).to_csv(
            'best_objective_{}.csv'.format(test_id), index=False)

        pd.DataFrame(self.population_fitness).to_csv(
            'last_population_fitness_{}.csv'.format(test_id), index=False)

        pd.DataFrame(self.generation_fitness).to_csv(
            'generation_fitness_{}.csv'.format(test_id), index=False)

        pd.DataFrame(self.best_solutions_fitness).to_csv(
            'best_solutions_fitness_{}.csv'.format(test_id), index=False)

    def initialize_population(self):
        """
        Initializes the population
        """
        self.pop_size = (self.population_size, self.num_variables)
        self.population = np.random.uniform(
            low=self.lb, high=self.ub, size=self.pop_size)
        self.initial_population = np.copy(self.population)

    def cal_pop_fitness(self, population):
        """
        Calculating the fitness values of all solutions in the current population.
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol_idx, sol in enumerate(population):
            fitness = self.fitness_func(sol, sol_idx)
            pop_fitness.append(self.scale(fitness))

        pop_fitness = np.array(pop_fitness)

        if(self.linear_scaling):
            pop_fitness = self.apply_linear_scale(pop_fitness)

        self.fitness_eval = self.fitness_eval + pop_fitness.shape[0]
        self.population_fitness = pop_fitness
        return pop_fitness

    def run(self, debug=False):
        print('Starting GA...')
        self.initialize_population()

        for generation in range(self.num_generations):
            if self.fitness_eval >= self.max_feval:
                break

            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness(self.population)
            best_fitness_index = np.argmax(fitness)
            if(fitness[best_fitness_index] > self.best_fitness):
                self.best_fitness = fitness[best_fitness_index]
                self.best_solution = self.population[best_fitness_index, :]

            # Selecting the best parents in the population for mating.
            prob = np.random.rand()
            if prob < 0.5:
                parents = self.encode(self.roulette_selection(
                    fitness, self.population_size))
            else:
                parents = self.encode(self.tournament_selection(
                    fitness, self.population_size, self.tournament_candidates))

            # Crossover
            offspring_crossover = self.single_point_crossover(
                parents, generation)

            # Mutation
            offspring_mutated = self.flip_bit_mutation_per_individual(
                offspring_crossover, generation)

            # Update population
            # Using generational approach, so population is the offspring only
            self.population = self.decode(offspring_mutated)

            # Store GA progress data
            self.generation_parents[generation] = parents
            self.generation_solutions[generation] = self.population
            self.generation_fitness.append(fitness)
            self.generation_fobj.append([self.remove_linear_scale(self.descale(
                x)) if self.linear_scaling else self.descale(x) for x in fitness])
            self.best_solutions_fitness.append(np.max(fitness))
            self.generation_offspring_mutated[generation] = offspring_mutated
            self.generation_offspring_crossover[generation] = offspring_crossover

        self.best_objective = self.remove_linear_scale(self.descale(
            self.best_fitness)) if self.linear_scaling else self.descale(self.best_fitness)

        print('Finishing GA...')

    def roulette_selection(self, fitness, num_parents):
        """
        Selects the parents using the roulette wheel selection technique.
        """
        fitness_sum = np.sum(fitness)
        probs = fitness / fitness_sum
        a = np.zeros(probs.shape[0])
        for i in range(probs.shape[0]):
            a[i] = probs[i] + np.sum(probs[:(i-1)]) if i > 0 else 0

        parents = np.empty((num_parents, self.num_variables))
        rand_probs = np.random.rand(num_parents)
        for parent_num in range(num_parents):
            rand_prob = rand_probs[parent_num]
            for idx in range(probs.shape[0]):
                if a[idx] > rand_prob:
                    parents[parent_num, :] = self.population[idx, :]
                    break

        return parents

    def tournament_selection(self, fitness, num_parents, num_candidates):
        """
        Selects the parents with tournament selection
        """
        parents = np.empty((num_parents, self.num_variables))
        winner_indexes = []
        nx = fitness.shape[0]

        for parent_num in range(num_parents):

            tournament_indices = np.random.randint(
                low=0.0, high=nx, size=num_candidates)
            tournament_fitness = fitness[tournament_indices]

            winner_index = tournament_indices[np.argsort(
                tournament_fitness)[-1]]
            winner_indexes.append(winner_index)
            parents[parent_num, :] = self.population[winner_index, :]

        return parents

    def single_point_crossover(self, parents, generation):
        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        """
        n = parents.shape[0]
        np.random.shuffle(parents)
        offspring = np.empty((n, self.num_variables), dtype=parents.dtype)

        for var_index in range(self.num_variables):
            offspring_list = []
            for index in np.arange(int(n), step=2):
                parent1 = parents[index, var_index]
                # Gets a random parent to mate
                parent2 = parents[index + 1, var_index]

                offspring1 = parent1
                offspring2 = parent2

                # Crossover probability
                prob = np.random.random()

                # Apply crossover between parents only if the probability is lower
                if(prob < self.get_crossover_probability(generation)):
                    crossover_point = np.random.randint(
                        low=0, high=self.num_bits)
                    offspring1 = parent1[:crossover_point] + \
                        parent2[crossover_point:]
                    offspring2 = parent2[:crossover_point] + \
                        parent1[crossover_point:]

                offspring_list.append(offspring1)
                offspring_list.append(offspring2)

            offspring[:, var_index] = np.array(offspring_list)

        return offspring

    def flip_bit_mutation_per_individual(self, offsprings, generation):
        num_offsprings = offsprings.shape[0]
        pm = self.get_mutation_probability(generation)
        for i in range(num_offsprings):
            for var_index in range(self.num_variables):
                prob = np.random.random()
                if prob < pm:
                    bits_to_flip = np.ceil(pm * self.num_bits)
                    rand_indexes = np.random.randint(
                        low=0, high=self.num_bits, size=int(bits_to_flip))

                    variable = str(offsprings[i, var_index])
                    for k in rand_indexes:
                        chromosome = bool(int(variable[k]))
                        mutated = not chromosome
                        variable = variable[:k] + str(int(mutated)) + (
                            variable[(k+1):] if k < (self.num_bits-1) else '')

                    offsprings[i, var_index] = variable

        return offsprings

    def flip_bit_mutation_per_bit(self, offsprings):
        num_offsprings = offsprings.shape[0]
        for i in range(num_offsprings):
            for var_index in range(self.num_variables):
                variable = ''
                for k in range(self.num_bits):
                    prob = np.random.random()
                    chromosome = bool(int(offsprings[i, var_index][k]))
                    mutated = not chromosome if prob < self.mutation_probability else chromosome
                    variable = variable + str(int(mutated))

                offsprings[i, var_index] = variable

        return offsprings

    def encode(self, x):
        nx = x.shape[0]
        encoded = []
        for var_index in range(self.num_variables):
            dx = self.precision[var_index]
            li = np.repeat(self.lb[var_index], nx)
            xint = ((x[:, var_index] - li)/dx).astype('int')
            encoded.append(self.int_to_binary_array(xint))

        encoded_matrix = np.transpose(np.array(encoded))
        return np.array(encoded_matrix)

    def int_to_binary_array(self, x):
        return np.array([self.binary_to_gray(np.binary_repr(value, self.num_bits)) for value in x]).flatten()

    def decode(self, x):
        nx = len(x)
        decoded = np.zeros((nx, self.num_variables))
        for i in range(nx):
            for var_index in range(self.num_variables):
                dx = self.precision[var_index]
                k_str = self.gray_to_binary(x[i][var_index])
                k_int = int(k_str, 2)
                decoded[i, var_index] = k_int * dx + self.lb[var_index]

        return decoded

    def binary_to_gray(self, b):
        # return b
        g = ''
        for index in range(len(b)):
            gray_value = b[index] if index == 0 else str(
                int(b[index-1]) ^ int(b[index]))
            g = g + gray_value
        return g

    def gray_to_binary(self, g):
        # return g
        b = ''
        for index in range(len(g)):
            bin_value = g[index] if index == 0 else str(
                int(b[index-1]) ^ int(g[index]))
            b = b + bin_value
        return b

    def scale(self, fx):
        """
        Scales the objective with 1/(fx+c) scaling
        """
        return 1/(fx + self.scale_factor)

    def descale(self, fitness):
        if(fitness == 0):
            return np.nan

        return (1 - fitness)/fitness

    def apply_linear_scale(self, fitness):
        mean_fitness = np.mean(fitness)
        max_fitness = np.max(fitness)

        if(mean_fitness == max_fitness):  # no linear scaling is necessary
            return fitness

        a, b = self.get_scaling_factors(
            mean_fitness, max_fitness, self.K_scaling)

        return np.maximum(np.zeros_like(fitness), fitness * a + b)

    def get_scaling_factors(self, mean_fitness, max_fitness, K):
        a = (K-1)*mean_fitness / (max_fitness - mean_fitness)
        b = (1 - a) * mean_fitness

        return a, b

    def remove_linear_scale(self, fitness):
        mean_fitness = np.mean(self.population_fitness)
        max_fitness = np.max(self.population_fitness)

        if(mean_fitness == max_fitness):  # no linear scaling is necessary
            return fitness

        a, b = self.get_scaling_factors(
            mean_fitness, max_fitness, self.K_scaling)

        return (fitness - b)/a

    def get_mutation_probability(self, generation):
        return self.min_mutation_probability - (self.max_mutation_probability - self.min_mutation_probability) * generation/self.num_generations

    def get_crossover_probability(self, generation):
        return self.min_crossover_probability + (self.max_crossover_probability - self.min_crossover_probability) * generation/self.num_generations
