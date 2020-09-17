import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, func, lb, ub, max_feval=10000, swarm_size=100, acceleration=[0.01, 0.01], constrition=1, inertia=0.7, topology='gbest'):
        self.max_feval = max_feval
        self.func = func
        self.swarm_size = swarm_size
        self.num_variables = len(lb)
        self.lb = lb
        self.ub = ub
        self.topology = topology
        self.x = constrition
        self.w = inertia
        self.c1, self.c2 = acceleration

        self.fevals = 0
        self.global_best_cost = np.Infinity
        self.personal_best_cost = np.ones(self.swarm_size) * np.Infinity
        self.personal_best_solution = np.zeros(
            (self.swarm_size, self.num_variables))
        self.iteration_personal_best_cost = []
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
        self.iteration_personal_best_cost.append(
            np.mean(self.personal_best_cost))
        self.iteration_average_cost.append(np.mean(swarm_cost))
        self.iteration_global_best_cost.append(self.global_best_cost)

        return swarm_cost

    def get_social_component_solution(self, swarm, swarm_cost):
        if(self.topology == 'gbest'):
            return np.array([self.global_best_solution] * self.swarm_size)
        elif(self.topology == 'lbest'):
            solution = np.zeros_like(swarm)
            for i in range(self.swarm_size):
                if(i == 0):
                    neighbors = np.vstack(
                        (swarm[-1, :], swarm[0, :], swarm[1, :]))
                    neighbors_cost = np.array(
                        [swarm_cost[-1], swarm_cost[0], swarm_cost[1]])
                    best_neighbor = np.argmin(neighbors_cost)
                elif(i == (self.swarm_size - 1)):
                    neighbors = np.vstack(
                        (swarm[-2, :], swarm[-1, :], swarm[0, :]))
                    neighbors_cost = np.array(
                        [swarm_cost[-2], swarm_cost[-1], swarm_cost[0]])
                    best_neighbor = np.argmin(neighbors_cost)
                else:
                    neighbors = np.vstack(
                        (swarm[i-1, :], swarm[i, :], swarm[i+1, :]))
                    neighbors_cost = np.array(
                        [swarm_cost[i-1], swarm_cost[i], swarm_cost[i+1]])
                    best_neighbor = np.argmin(neighbors_cost)

                solution[i, :] = neighbors[best_neighbor, :]

                # if(neighbors_cost[best_neighbor] < swarm_cost[i]):
                #     solution[i, :] = neighbors[best_neighbor]
                # else:
                #     solution[i, :] = swarm[i]
            return solution

    def update_particles_velocity(self, swarm, swarm_cost):
        updated_velocities = []
        updated_swarm = []

        social_best_sol = self.get_social_component_solution(swarm, swarm_cost)

        for i, particle in enumerate(swarm):
            r1 = np.random.random(size=self.num_variables)
            r2 = np.random.random(size=self.num_variables)

            cognitive_component = self.c1 * r1 * \
                (self.personal_best_solution[i, :] - particle)

            social_component = self.c2 * r2 * \
                (social_best_sol[i, :] - particle)

            velocity = self.w * self.swarm_velocity[i, :] + \
                cognitive_component + social_component

            velocity = self.x * velocity

            updated_velocities.append(velocity)
            updated_swarm.append(particle + velocity)

        self.swarm_velocity = np.array(updated_velocities)
        self.swarm = np.array(updated_swarm)

    def run(self, debug=True):
        self.initialize_swarm()
        while(self.fevals < self.max_feval):
            # Calculating the cost of each particle in the swarm.
            self.evaluate_swarm_cost_function(self.swarm)
            self.update_particles_velocity(self.swarm, self.swarm_cost)

        if(debug):
            self.report()

        return self.global_best_solution, self.global_best_cost

    def report(self):
        print('Best global particle: {}'.format(self.global_best_solution))
        print('Best global cost: {}'.format(self.global_best_cost))
        self.plot_charts()

    def plot_charts(self):
        _, ax = plt.subplots(3, 1)
        ax[0].plot(self.iteration_personal_best_cost)
        ax[1].plot(self.iteration_average_cost)
        ax[2].plot(self.iteration_global_best_cost)
        # plt.legend([''])
        plt.tight_layout()
        plt.show()
