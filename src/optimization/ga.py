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

        self.fitness_eval = 0

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
        print('Avg fitness: {}'.format(np.mean(pop_fitness)))
        return pop_fitness

    def run(self):
        print('Starting GA...')
        self.initialize_population()

        for generation in range(self.num_generations):
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
            offspring_mutated = self.flip_bit_mutation(offspring_crossover)

            # Survivor selection
            # offspring_survived = self.survivor_selection(
            #     offspring_mutated, self.population_size)

            # Store GA progress data
            self.generation_parents[generation] = parents
            self.generation_solutions[generation] = self.population
            self.generation_fitness.append(fitness)
            self.best_solutions_fitness.append(np.max(fitness))
            self.generation_offspring_mutated[generation] = offspring_mutated
            self.generation_offspring_crossover[generation] = offspring_crossover

            # Log generation results:
            print('Generation #{}: Best fitness: {}; Avg Fitness: {}; Worst Fitness: {}'.format(
                generation, np.max(fitness), np.mean(fitness), np.min(fitness)))
            print('Generation #{}: Best solution: {}'.format(
                generation, self.population[np.argmax(fitness)]))
            print('Generation #{}: Worst solution: {}'.format(
                generation, self.population[np.argmin(fitness)]))
            print(
                '===========================================================================================')
            # Update population
            self.population = self.decode(offspring_mutated)

        print('Finishing GA...')

    # def survivor_selection_rank(self, parents, fitness_parents, offsprings, num_survivors):
    #     """
    #     Selects the survivors based on rank
    #     """
    #     filtered_offsprings = []
    #     for offspring in offsprings:
    #         if offspring not in parents:
    #             filtered_offsprings.append(offspring)

    #     fitness = self.cal_pop_fitness(
    #         self.decode(np.array(filtered_offsprings)))
    #     fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    #     fitness_sorted.reverse()

    #     parents = np.empty((num_survivors, self.population.shape[1]))
    #     for parent_num in range(num_survivors):
    #         parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]
    #     return parents

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

    def random_selection(self, fitness, num_parents):
        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        """
        parents = np.empty((num_parents, self.population.shape[1]))

        rand_indices = np.random.randint(
            low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :]
        return parents

    def single_point_crossover(self, parents):
        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        """
        n = parents.shape[0]

        pairs = list(itertools.combinations(range(0, n), 2))
        # two offsprings for each pair
        num_offsprings = len(pairs) * 2
        offspring = np.empty(
            (num_offsprings, self.num_variables), dtype=parents.dtype)

        for var_index in range(self.num_variables):
            offspring_list = []
            for index_1, index_2 in pairs:
                parent1 = parents[index_1, var_index]
                parent2 = parents[index_2, var_index]

                # Crossover probability
                prob = np.random.random()

                offspring1 = parent1
                offspring2 = parent2

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

    def flip_bit_mutation(self, offsprings):
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
        g = ''
        for index in range(len(b)):
            gray_value = b[index] if index == 0 else str(
                int(b[index-1]) ^ int(b[index]))
            g = g + gray_value
        return g

    def gray_to_binary(self, g):
        b = ''
        for index in range(len(g)):
            bin_value = g[index] if index == 0 else str(
                int(b[index-1]) ^ int(g[index]))
            b = b + bin_value
        return b
