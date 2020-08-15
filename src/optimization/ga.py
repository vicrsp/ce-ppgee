import numpy as np
import itertools
import random
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
    def __init__(self, lb, ub, fitness_func, pop_size=10, num_generations=10, crossover_probability=0.6, mutation_probability=0.05, num_bits=20, tournament_candidates=10):
        self.population_size = pop_size
        self.num_variables = len(lb)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.lb = lb
        self.ub = ub
        self.num_generations = num_generations
        self.fitness_func = fitness_func
        self.num_bits = num_bits  # number of bits to encode each gene
        self.tournament_candidates = tournament_candidates

        self.precision = (np.array(self.ub) - np.array(self.lb)
                          )/(np.exp2(self.num_bits) - 1)
        self.best_solutions_fitness = []
        self.generation_fitness = []
        self.generation_solutions = {}
        self.generation_parents = {}
        self.generation_offspring_mutated = {}
        self.generation_offspring_crossover = {}

        self.x_tol = 10 ** -6
        self.fitness_tol = 10 ** -6
        self.fitness_eval = 0
        self.max_feval = 10000

        print(self.precision)

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
            pop_fitness.append(fitness)

        pop_fitness = np.array(pop_fitness)

        self.fitness_eval = self.fitness_eval + pop_fitness.shape[0]
        return pop_fitness

    def run(self, debug=False):
        print('Starting GA...')
        self.initialize_population()

        for generation in range(self.num_generations):
            if self.fitness_eval >= self.max_feval:
                break

            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness(self.population)

            # Selecting the best parents in the population for mating.
            prob = np.random.rand()
            if prob < 0.5:
                parents = self.encode(self.roulette_selection(
                    fitness, self.population_size))
            else:
                parents = self.encode(self.tournament_selection(
                    fitness, self.population_size, self.tournament_candidates, with_replacement=True))

            # Crossover
            offspring_crossover = self.single_point_crossover(parents)

            # Mutation
            offspring_mutated = self.flip_bit_mutation_per_individual(
                offspring_crossover)

            # Update population
            # Using generational approach, so population is the offspring only
            self.population = self.decode(offspring_mutated)

            # Store GA progress data
            self.generation_parents[generation] = parents
            self.generation_solutions[generation] = self.population
            self.generation_fitness.append(fitness)
            self.best_solutions_fitness.append(np.max(fitness))
            self.generation_offspring_mutated[generation] = offspring_mutated
            self.generation_offspring_crossover[generation] = offspring_crossover

            # if(debug):
            #     # Log generation results:
            #     print('Generation #{}: Best fitness: {}; Avg Fitness: {}; Worst Fitness: {}'.format(
            #         generation, np.max(fitness), np.mean(fitness), np.min(fitness)))
            #     print('Generation #{}: Best solution: {}'.format(
            #         generation, self.population[np.argmax(fitness)]))
            #     print('Generation #{}: Worst solution: {}'.format(
            #         generation, self.population[np.argmin(fitness)]))
            #     print(
            #         '===========================================================================================')

            # if(generation > 0):
            #     generation_best = self.population[np.argmax(fitness)]
            #     previous_best = self.generation_solutions[generation - 1][np.argmax(
            #         self.best_solutions_fitness[generation-1]), :]

            #     diff_best = np.linalg.norm(generation_best-previous_best)

            #     if diff_best < self.x_tol:
            #         print('Terminating due to BEST x_tol convergence...')
            #         break
            # if(generation > 0):
            #     previous_generation = self.generation_solutions[generation - 1]
            #     diff_best = np.linalg.norm(self.population-previous_generation)

            #     if diff_best < self.x_tol:
            #         print('Terminating due AVG to x_tol convergence...')
            #         break

            # if(generation > 0):
            #     generation_best = np.max(fitness)
            #     previous_best = self.best_solutions_fitness[generation - 1]

            #     diff = np.abs(generation_best -
            #                   previous_best)/generation_best
            #     if diff < self.fitness_tol:
            #         print('Terminating due to BEST fitness_tol convergence...')
            #         break

            # if(generation > 0):
            #     generation_mean = np.mean(fitness)
            #     previous_mean = np.mean(
            #         self.generation_fitness[generation - 1])

            #     diff = np.abs(generation_mean -
            #                   previous_mean)/generation_mean

            #     if diff < self.fitness_tol:
            #         print('Terminating due to AVG fitness_tol convergence...')
            #         break

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

    def tournament_selection(self, fitness, num_parents, num_candidates, with_replacement=True):
        """
        Selects the parents with tournament selection
        """
        parents = np.empty((num_parents, self.num_variables))
        winner_indexes = []
        nx = fitness.shape[0]

        for parent_num in range(num_parents):
            tournament_fitness = []
            tournament_indices = []

            if with_replacement:
                tournament_indices = np.random.randint(
                    low=0.0, high=nx, size=num_candidates)
                tournament_fitness = fitness[tournament_indices]
            else:
                rand_indices = np.random.randint(
                    low=0.0, high=(nx - parent_num), size=num_candidates)
                indexes_remaining = np.where(range(nx) not in winner_indexes)

                tournament_fitness = fitness[indexes_remaining]
                tournament_indices = indexes_remaining[rand_indices]

            winner_index = tournament_indices[np.argsort(
                tournament_fitness)[-1]]
            winner_indexes.append(winner_index)
            parents[parent_num, :] = self.population[winner_index, :]

        return parents

    def single_point_crossover(self, parents):
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
                if(prob < self.crossover_probability):
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

    def flip_bit_mutation_per_individual(self, offsprings):
        num_offsprings = offsprings.shape[0]
        for i in range(num_offsprings):
            for var_index in range(self.num_variables):
                prob = np.random.random()
                if prob < self.mutation_probability:
                    bits_to_flip = np.ceil(
                        self.mutation_probability * self.num_bits)
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
