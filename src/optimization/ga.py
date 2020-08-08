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
    def __init__(self, lb, ub, fitness_func, pop_size=10, num_generations=10, num_genes=2, crossover_probability=0.6, mutation_probability=0.05, num_bits=20):
        self.population_size = pop_size
        self.num_genes = num_genes
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.lb = lb
        self.ub = ub
        self.num_generations = num_generations
        self.fitness_func = fitness_func
        self.num_bits = num_bits  # number of bits to encode each gene

        self.precision = (np.array(self.ub) - np.array(self.lb)
                          )/(np.exp2(self.num_bits) - 1)
        self.best_solutions_fitness = []
        self.fitness_eval = 0

    def initialize_population(self):
        """
        Initializes the population
        """
        self.pop_size = (self.population_size, self.num_genes)
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
        print('Fitness evals: {}'.format(self.fitness_eval))
        return pop_fitness

    def run(self):
        print('Starting GA...')
        self.initialize_population()

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness(self.population)
            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(np.max(fitness))

            # Selecting the best parents in the population for mating.
            parents = self.encode(self.roulette_selection(
                fitness, self.population_size))

            # Crossover
            offspring_crossover = self.single_point_crossover(parents)

            # Mutation
            offspring_mutated = self.flip_bit_mutation(offspring_crossover)

            # Survivor selection
            # offspring_survived = self.survivor_selection(
            #     offspring_mutated, self.population_size)

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
        # An array holding the start values of the ranges of probabilities.
        probs_start = np.zeros(probs.shape, dtype=np.float)
        # An array holding the end values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape, dtype=np.float)

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, self.num_genes))
        for parent_num in range(num_parents):
            rand_prob = np.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :]
                    break
        return parents

    def tournament_selection(self, fitness, num_parents):
        """
        Selects the parents with roulette selection
        """

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
        # combination of 2
        num_offsprings = (factorial(n) // factorial(2) // factorial(n-2))
        offspring = np.empty(
            (num_offsprings * self.num_genes, self.num_genes), dtype=parents.dtype)

        #pairs = list(itertools.combinations(range(0, n), 2))

        for gene in range(self.num_genes):
            parents_pairs = list(itertools.combinations(parents[:, gene], 2))
            random.shuffle(parents_pairs)

            offspring_list = []
            for i in range(len(pairs)):
                parent1 = parents_pairs[i][0]
                parent2 = parents_pairs[i][1]

                # Crossover probability
                prob = np.random.random()

                offspring1 = parent1
                offspring2 = parent2

                # Apply crossover between parents only if the probability is
                if(prob < self.crossover_probability):
                    crossover_point = np.random.randint(
                        low=0, high=self.num_bits)
                    offspring1 = parent1[:crossover_point] + \
                        parent2[crossover_point:]
                    offspring2 = parent2[:crossover_point] + \
                        parent1[crossover_point:]

                offspring_list.append(offspring1)
                offspring_list.append(offspring2)

            offspring[:, gene] = np.array(offspring_list)

        return offspring

    def flip_bit_mutation(self, offsprings):
        num_offsprings = offsprings.shape[0]
        for i in range(num_offsprings):
            for gene in range(self.num_genes):
                variable = ''
                for k in range(self.num_bits):
                    prob = np.random.random()
                    chromosome = bool(int(offsprings[i, gene][k]))
                    mutated = not chromosome if prob < self.mutation_probability else chromosome
                    variable = variable + str(int(mutated))

                offsprings[i, gene] = variable

        return offsprings

    def encode(self, x):
        nx = x.shape[0]
        encoded = []
        for gene in range(self.num_genes):
            dx = self.precision[gene]
            li = np.repeat(self.lb[gene], nx)
            xint = ((x[:, gene] - li)/dx).astype('int')
            encoded.append(self.int_to_binary_array(xint))

        encoded_matrix = np.transpose(np.array(encoded))
        return np.array(encoded_matrix)

    def int_to_binary_array(self, x):
        return np.array([np.binary_repr(value, self.num_bits) for value in x]).flatten()

    def decode(self, x):
        nx = len(x)
        decoded = np.zeros((nx, self.num_genes))
        for i in range(nx):
            for gene in range(self.num_genes):
                dx = self.precision[gene]
                k_str = x[i][gene]
                k_int = int(k_str, 2)
                decoded[i, gene] = k_int * dx + self.lb[gene]

        return decoded
