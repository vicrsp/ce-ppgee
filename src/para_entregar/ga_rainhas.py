from math import factorial
import numpy as np
import itertools
import random


class GAPermutation:
    def __init__(self, fitness_func, pop_size=100, num_generations=300, max_int=8, crossover_probability=0.6, mutation_probability=0.05, use_inversion_mutation=False):
        self.population_size = pop_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.num_generations = num_generations
        self.fitness_func = fitness_func
        self.max_int = max_int
        self.use_inversion_mutation = use_inversion_mutation

        self.fitness_eval = 0
        self.scale_factor = self.max_int*(self.max_int-1)/2
        self.best_objective = np.Infinity
        self.best_solution = []
        self.best_fitness = 0
        self.converged = False

    def initialize_population(self):
        """
        Initializes the population
        """
        self.pop_size = (self.population_size, self.max_int)
        # self.population = np.random.randint(
        #     low=0, high=self.max_int, size=self.pop_size)
        self.population = np.zeros(self.pop_size)
        for i in range(self.population_size):
            array = np.arange(self.max_int)
            np.random.shuffle(array)
            self.population[i, :] = array

        self.initial_population = np.copy(self.population)

    def cal_pop_fitness(self, population):
        """
        Calculating the fitness values of all solutions in the current population.
        """
        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol in population:
            fitness = self.scale(self.fitness_func(sol))
            pop_fitness.append(fitness)

        pop_fitness = np.array(pop_fitness)

        self.fitness_eval = self.fitness_eval + pop_fitness.shape[0]
        self.population_fitness = pop_fitness

        return pop_fitness

    def run(self):
        self.initialize_population()

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness(self.population)
            best_fitness_index = np.argmax(fitness)
            if(fitness[best_fitness_index] > self.best_fitness):
                self.best_fitness = fitness[best_fitness_index]
                self.best_solution = self.population[best_fitness_index, :]

            if self.descale(self.best_fitness) == 0:
                self.converged = True
                break

            # Selecting the best parents in the population for mating.
            parents = self.selection(fitness, int(self.population_size/2))

            # Crossover
            offspring_crossover = self.crossover(parents)

            # Mutation
            offspring_mutated = self.mutation(offspring_crossover)

            # Survivor selection
            offspring_survived = self.survivor_selection(
                fitness, offspring_mutated)

            # Update population
            self.population = offspring_survived

        self.best_objective = self.descale(self.best_fitness)

    def crossover(self, parents):
        prob = np.random.rand()
        offspring = parents
        if prob < self.get_crossover_probability():
            offspring = self.ordered_crossover(parents)

        return offspring

    def stochastic_universal_sampling_selection(self, fitness, num_parents):
        """
        Selects the parents using SUS selection technique.
        """
        sorted_parents = self.population[np.flip(np.argsort(fitness))]
        sorted_fitness = fitness[np.flip(np.argsort(fitness))]

        fitness_sum = np.sum(fitness)

        distance = fitness_sum / float(num_parents)
        start = random.uniform(0, distance)
        points = [start + i*distance for i in range(num_parents)]

        parents = np.empty((num_parents, self.max_int))
        parents_fitness = np.empty(num_parents)

        parent_num = 0
        for p in points:
            idx = 0
            r = sorted_fitness[idx]
            while r < p:
                idx = idx + 1
                r = r + sorted_fitness[idx]

            parents[parent_num, :] = sorted_parents[idx, :]
            parents_fitness[parent_num] = sorted_fitness[idx]
            parent_num = parent_num + 1
        return parents, parents_fitness

    def selection(self, fitness, num_tournament):
        parents = np.zeros((2, self.max_int))

        parent_selection, parent_fitness = self.stochastic_universal_sampling_selection(
            fitness, 2)

        sort_indexes = np.argsort(parent_fitness)
        best = parent_selection[sort_indexes[-1], :]
        second_best = parent_selection[sort_indexes[-2], :]

        parents[0, :] = best
        parents[1, :] = second_best

        return parents

    def survivor_selection(self, fitness, offspring):
        offspring_fitness = self.cal_pop_fitness(offspring)
        pop_fitness = np.hstack((fitness, offspring_fitness))
        merged_pop = np.vstack((self.population, offspring))

        sort_indexes = np.argsort(pop_fitness)
        sorted_pop = merged_pop[sort_indexes]
        return sorted_pop[2:, :]

    def inversion_mutation(self, offsprings):
        m, n = offsprings.shape
        mutated = np.zeros(offsprings.shape)
        for i in range(m):
            prob = np.random.rand()
            mutated[i, :] = offsprings[i, :]
            if prob < self.get_mutation_probability():
                pos_1, pos_2 = np.sort(
                    np.random.randint(low=0, high=n, size=2))
                flipped_array = np.flip(offsprings[i, pos_1:pos_2])
                mutated[i, pos_1:pos_2] = flipped_array

        return mutated

    def mutation(self, offsprings):
        if(self.use_inversion_mutation):
            return self.inversion_mutation(offsprings)
        else:
            return self.swap_mutation(offsprings)

    def swap_mutation(self, offsprings):
        m, n = offsprings.shape

        mutated = np.zeros(offsprings.shape)
        for i in range(m):
            prob = np.random.rand()
            mutated[i, :] = offsprings[i, :]
            if prob < self.get_mutation_probability():
                pos_1, pos_2 = np.random.randint(low=0, high=n, size=2)
                first_num = offsprings[i, pos_1]
                second_num = offsprings[i, pos_2]
                mutated[i, pos_1] = second_num
                mutated[i, pos_2] = first_num

        return mutated

    def ordered_crossover(self, parents):
        """
        Executes an ordered crossover (OX) on the input individuals.
        """
        parent1, parent2 = parents[0, :], parents[1, :]

        size = len(parent1)
        a, b = random.sample(range(size), 2)
        if a > b:
            a, b = b, a

        holes1, holes2 = [True] * size, [True] * size
        for i in range(size):
            if i < a or i > b:
                holes1[int(parent2[i])] = False
                holes2[int(parent1[i])] = False

        # We must keep the original values somewhere before scrambling everything
        temp1, temp2 = parent1, parent2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if not holes1[int(temp1[(i + b + 1) % size])]:
                parent1[int(k1 % size)] = temp1[int((i + b + 1) % size)]
                k1 += 1

            if not holes2[int(temp2[(i + b + 1) % size])]:
                parent2[int(k2 % size)] = temp2[int((i + b + 1) % size)]
                k2 += 1

        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            parent1[i], parent2[i] = parent2[i], parent1[i]

        return np.array([parent1, parent2])

    def scale(self, fx):
        """
        Scales the objective with Cmax scaling
        """
        return self.scale_factor - fx

    def descale(self, fitness):
        return self.scale_factor - fitness

    def get_mutation_probability(self):
        return self.mutation_probability

    def get_crossover_probability(self):
        return self.crossover_probability
