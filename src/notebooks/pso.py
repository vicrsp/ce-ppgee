import numpy as np


class PSO:
    def __init__(self, func, lb, ub, topology='best', constrition=1, inertia=1, acceleration=[1, 1], swarm_size=100, max_feval=1000):
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
        self.local_best_cost = np.Infinity

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
        self.local_best_cost = np.Infinity
        for sol in swarm:
            cost = self.func(sol)
            swarm_cost.append(cost)

            # Update global best cost
            if(cost < self.global_best_cost):
                self.global_best_cost = cost
                self.global_best_solution = sol

            # Update local best cost
            if(cost < self.local_best_cost):
                self.local_best_cost = cost
                self.local_best_solution = sol

        swarm_cost = np.array(swarm_cost)

        self.fevals = self.fevals + swarm_cost.shape[0]
        self.swarm_cost = swarm_cost
        return swarm_cost

    def update_particles_velocity(self, swarm):
        updated_velocities = []
        updated_swarm = []
        for i, particle in enumerate(swarm):
            r1 = np.random.random(size=self.num_variables)
            r2 = np.random.random(size=self.num_variables)

            cognitive_component = self.c1 * r1 * \
                (self.local_best_solution - particle)

            social_component = self.c2 * r2 * \
                (self.global_best_solution - particle)

            velocity = self.w * self.swarm_velocity[i, :] + \
                cognitive_component + social_component

            velocity = self.x * velocity

            updated_velocities.append(velocity)
            updated_swarm.append(particle + velocity)

        self.swarm_velocity = np.array(updated_velocities)
        self.swarm = np.array(updated_swarm)

    def run(self, debug=False):
        self.initialize_swarm()
        while(self.fevals < self.max_feval):
            # Calculating the cost of each particle in the swarm.
            self.evaluate_swarm_cost_function(self.swarm)
            self.update_particles_velocity(self.swarm)

            if(debug):
                self.report()
                print('Best local particle: {}'.format(
                    self.local_best_solution))
                print('Best local cost: {}'.format(self.local_best_cost))

        self.report()

    def report(self):
        print('Best global particle: {}'.format(self.global_best_solution))
        print('Best global cost: {}'.format(self.global_best_cost))

    def plot_charts(self):
