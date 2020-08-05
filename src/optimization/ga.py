import numpy as np


class GA:
    def __init__(self, lb, ub, fitness_func, pop_size=10, num_generations=10, n_genes=2, crossover_probability=0.6, p_mutation=0.1):
        self.population_size = pop_size
        self.num_genes = n_genes
        self.crossover_probability = crossover_probability
        self.p_mutation = p_mutation
        self.lb = lb
        self.ub = ub
        self.num_generations = num_generations
        self.fitness_func = fitness_func

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
            parents = self.random_selection(fitness, 5)

            # Crossover
            offspring_crossover = self.single_point_crossover(
                parents, self.pop_size)
            print(parents)

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

        offspring = np.empty(offspring_size)

        for k in range(offspring_size[0]):
            # The point at which crossover takes place between two parents. Usually, it is at the center.
            crossover_point = np.random.randint(
                low=0, high=parents.shape[1], size=1)[0]

            if self.crossover_probability != None:
                probs = np.random.random(size=parents.shape[1])
                indices = np.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = np.random.sample(set(indices), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # The new offspring has its first half of its genes from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx,
                                                      0:crossover_point]
            # The new offspring has its second half of its genes from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx,
                                                     crossover_point:]
        return offspring

    def encode(self, x):
        return self.real_to_binary(x)

    def decode(self, x):
        return x

    def real_to_binary(self, x):
        return x
