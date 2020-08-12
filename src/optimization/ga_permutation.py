import numpy as np
import itertools
import random
from math import factorial


class GAPermutation:
    def __init__(self, fitness_func, pop_size=10, num_generations=10, max_int=8, crossover_probability=0.6, mutation_probability=0.05):
        self.population_size = pop_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.num_generations = num_generations
        self.fitness_func = fitness_func
        self.max_int = max_int

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
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol in population:
            fitness = self.fitness_func(sol)
            pop_fitness.append(fitness)

        pop_fitness = np.array(pop_fitness)

        self.fitness_eval = self.fitness_eval + pop_fitness.shape[0]
        return pop_fitness

    def run(self, debug=True):
        print('Starting GAPermutation...')
        self.initialize_population()

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness(self.population)

            # Selecting the best parents in the population for mating.
            parents = self.selection(fitness, int(self.population_size/2))

            # Crossover
            offspring_crossover = self.crossover(parents)

            # Mutation
            offspring_mutated = self.mutation(offspring_crossover)

            # Survivor selection
            offspring_survived = self.survivor_selection(
                fitness, offspring_mutated)

            # Store GA progress data
            self.generation_parents[generation] = parents
            self.generation_solutions[generation] = self.population
            self.generation_fitness.append(fitness)
            self.best_solutions_fitness.append(np.max(fitness))
            self.generation_offspring_mutated[generation] = offspring_mutated
            self.generation_offspring_crossover[generation] = offspring_crossover

            if(debug):
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
            self.population = offspring_survived

        print('Finishing GA...')

    def crossover(self, parents):
        prob = np.random.rand()
        offspring = parents
        if prob < self.crossover_probability:
            offspring = self.cut_and_crossfill_crossover(parents)

        return offspring

    def tournament_selection(self, fitness, num_parents, num_candidates, with_replacement=True):
        """
        Selects the parents with tournament selection
        """
        parents = np.empty((num_parents, self.max_int))
        parents_fitness = np.empty(num_parents)

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
            parents_fitness[parent_num] = fitness[winner_index]

        return parents, parents_fitness

    def roulette_selection(self, fitness, num_parents):
        """
        Selects the parents using the roulette wheel selection technique.
        """
        fitness_sum = np.sum(fitness)
        probs = fitness / fitness_sum
        a = np.zeros(probs.shape[0])
        for i in range(probs.shape[0]):
            a[i] = probs[i] + np.sum(probs[:(i-1)]) if i > 0 else 0

        parents = np.empty((num_parents, self.max_int))
        parents_fitness = np.empty(num_parents)

        rand_probs = np.random.rand(num_parents)
        for parent_num in range(num_parents):
            rand_prob = rand_probs[parent_num]
            for idx in range(probs.shape[0]):
                if a[idx] > rand_prob:
                    parents[parent_num, :] = self.population[idx, :]
                    parents_fitness[parent_num] = fitness[idx]
                    break

        return parents, parents_fitness

    def selection(self, fitness, num_tournament):
        parents = np.zeros((2, self.max_int))

        # # Do not allow repeated individuals to be selected
        # random_indexes = np.arange(self.population_size)
        # np.random.shuffle(random_indexes)
        # random_indexes = random_indexes[:num_tournament]

        # random_fitness = fitness[random_indexes]
        # random_selection = self.population[random_indexes]
        # parent_selection, parent_fitness = self.roulette_selection(
        #     fitness, self.population_size)

        parent_selection, parent_fitness = self.tournament_selection(
            fitness, self.population_size, int(0.8 * self.population_size))

        sort_indexes = np.argsort(parent_fitness)
        best = parent_selection[sort_indexes[-1], :]
        second_best = parent_selection[sort_indexes[-2], :]

        # Do not allow selected parents to be the same
        for index in range(3, self.population_size):
            if (second_best == best).all():
                second_best = parent_selection[sort_indexes[-index], :]
            else:
                break

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

    def cut_and_crossfill_crossover(self, parents):
        """
        Applies CutAndCrossfill_Crossover.m. parents: array(2,N) => [parent1; parent2]
        """
        N = parents.shape[1]
        offspring = np.zeros((2, N))
        # single point crossover
        pos = int(np.floor(N*np.random.rand()))
        offspring[0, :pos] = parents[0, :pos]
        offspring[1, :pos] = parents[1, :pos]
        s1 = pos
        s2 = pos

        # if pos = N-1, no crossover happened
        if s1 < N:
            for i in range(N):
                check1 = 0
                check2 = 0
                for j in range(pos):
                    if (parents[1, i] == offspring[0, j]):
                        check1 = 1

                    if (parents[0, i] == offspring[1, j]):
                        check2 = 1

                if check1 == 0:
                    offspring[0, s1] = parents[1, i]
                    s1 = s1 + 1

                if check2 == 0:
                    offspring[1, s2] = parents[0, i]
                    s2 = s2 + 1

        return offspring

    def mutation(self, offsprings):
        m, n = offsprings.shape

        mutated = np.zeros(offsprings.shape)
        for i in range(m):
            prob = np.random.rand()
            mutated[i, :] = offsprings[i, :]
            if prob < self.mutation_probability:
                pos_1, pos_2 = np.random.randint(low=0, high=n, size=2)
                first_num = offsprings[i, pos_1]
                second_num = offsprings[i, pos_2]
                mutated[i, pos_1] = second_num
                mutated[i, pos_2] = first_num

        return mutated
