import numpy as np
import itertools
import random


class GA:
    def __init__(self, lb, ub, fitness_func, pop_size=10, num_generations=10, n_genes=2, crossover_probability=0.6, mutation_probability=0.05, num_bits=20):
        self.population_size = pop_size
        self.num_genes = n_genes
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

    def initialize_population(self):
        """
        Initializes the population
        """
        self.pop_size = (self.population_size, self.num_genes)
        self.population = np.random.uniform(
            low=self.lb, high=self.ub, size=self.pop_size)
        self.initial_population = np.copy(self.population)

    def cal_pop_fitness(self):
        """
        Calculating the fitness values of all solutions in the current population.
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol_idx, sol in enumerate(self.population):
            fitness = self.fitness_func(sol, sol_idx)
            pop_fitness.append(fitness)

        pop_fitness = np.array(pop_fitness)

        return pop_fitness

    def run(self):
        print('Starting optimizer...')
        self.initialize_population()

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness()
            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(np.max(fitness))

            # Selecting the best parents in the population for mating.
            parents = self.encode(self.random_selection(
                fitness, self.population_size))

            # Crossover
            offspring_crossover = self.single_point_crossover(
                parents, self.population_size)
            # Mutation
            offspring_mutated = self.flip_bit_mutation(offspring_crossover)
            # Update population
            self.population = self.decode(offspring_mutated)

    def random_selection(self, fitness, num_parents):
        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """
        parents = np.empty((num_parents, self.population.shape[1]))

        rand_indices = np.random.randint(
            low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :]
        return parents

    def single_point_crossover(self, parents, offspring_size):
        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        offspring = []
        parents_pairs = list(itertools.combinations(parents, 2))
        random.shuffle(parents_pairs)

        for pair in parents_pairs[:int(offspring_size/2)]:
            offspring1 = ''
            offspring2 = ''
            for gene in range(self.num_genes):
                parent1 = pair[0][gene*self.num_bits:(gene+1)*self.num_bits]
                parent2 = pair[1][gene*self.num_bits:(gene+1)*self.num_bits]

                # Crossover probability
                prob = np.random.random()

                # Apply crossover between parents only if the probability is
                if(prob < self.crossover_probability):
                    crossover_point = np.random.randint(
                        low=0, high=self.num_bits)
                    child1 = parent1[:crossover_point] + \
                        parent2[crossover_point:]
                    child2 = parent2[:crossover_point] + \
                        parent1[crossover_point:]

                    offspring1 = offspring1 + child1
                    offspring2 = offspring2 + child2
                else:
                    offspring1 = offspring1 + parent1
                    offspring2 = offspring2 + parent2

            offspring.append(offspring1)
            offspring.append(offspring2)

        return np.array(offspring).flatten()

    def flip_bit_mutation(self, offsprings):
        for offspring in offsprings:
            for gene in range(self.num_genes):
                variable = offspring[gene *
                                     self.num_bits:(gene+1)*self.num_bits]
                for chromossome in variable:
                    # mutation probability
                    prob = np.random.random()
                    if prob < self.mutation_probability:
                        chromossome = str((not bool(chromossome))*1)

        return offsprings

    def encode(self, x):
        nx = x.shape[0]
        encoded = []
        encoded_genes = []

        for gene in range(self.num_genes):
            dx = self.precision[gene]
            li = np.repeat(self.lb[gene], nx)
            xint = ((x[:, gene] - li)/dx).astype('int')
            encoded.append(self.int_to_binary_array(xint))

        encoded_matrix = np.transpose(np.array(encoded))
        for i in range(nx):
            value = ''
            for k in range(self.num_genes):
                value = value + str(encoded_matrix[i][k])

            encoded_genes.append(value)

        return np.array(encoded_genes)

    def int_to_binary_array(self, x):
        return np.array([np.binary_repr(value, self.num_bits) for value in x]).flatten()

    def decode(self, x):
        nx = len(x)
        decoded = np.zeros((nx, self.num_genes))
        for i in range(nx):
            for gene in range(self.num_genes):
                dx = self.precision[gene]
                k_str = x[i][gene*self.num_bits:(gene+1)*self.num_bits]
                k_int = int(k_str, 2)
                decoded[i, gene] = k_int * dx + self.lb[gene]

        return decoded
