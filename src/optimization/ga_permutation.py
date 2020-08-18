import numpy as np
import itertools
import random


class GAPermutation:
    def __init__(self, fitness_func, pop_size=10, num_generations=100, max_int=8, crossover_probability=0.6, mutation_probability=0.05):
        self.population_size = pop_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.num_generations = num_generations
        self.fitness_func = fitness_func
        self.max_int = max_int

        self.best_solutions_fitness = []
        self.generation_fitness = []
        self.generation_solutions = {}

        self.fitness_eval = 0
        self.scale_factor = self.max_int*(self.max_int-1)/2
        self.best_objective = np.Infinity
        self.best_solution = []
        self.best_fitness = 0

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
            fitness = self.scale(self.fitness_func(sol))
            pop_fitness.append(fitness)

        pop_fitness = np.array(pop_fitness)

        self.fitness_eval = self.fitness_eval + pop_fitness.shape[0]
        self.population_fitness = pop_fitness

        return pop_fitness

    def run(self, debug=True):
        print('Starting GAPermutation...')
        self.initialize_population()

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness(self.population)
            best_fitness_index = np.argmax(fitness)
            if(fitness[best_fitness_index] > self.best_fitness):
                self.best_fitness = fitness[best_fitness_index]
                self.best_solution = self.population[best_fitness_index, :]

            # Store GA progress data
            self.generation_solutions[generation] = self.population
            self.generation_fitness.append(fitness)
            self.best_solutions_fitness.append(np.max(fitness))

            if self.descale(self.best_fitness) == 0:
                print('Optimal solution found...')
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

        self.best_objective = self.descale(self.best_fitness)

        print('Finishing GA...')

    def crossover(self, parents):
        prob = np.random.rand()
        offspring = parents
        if prob < self.get_crossover_probability():
            offspring = self.ordered_crossover(parents)

        return offspring

    def tournament_selection(self, fitness, num_parents, num_candidates):
        """
        Selects the parents with tournament selection
        """
        parents = np.empty((num_parents, self.max_int))
        parents_fitness = np.empty(num_parents)

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

        parent_selection, parent_fitness = self.tournament_selection(
            fitness, self.population_size, int(0.8 * self.population_size))

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
