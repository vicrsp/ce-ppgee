import numpy as np
from src.optimization.ga import GA
from src.optimization.ga_permutation import GAPermutation
from src.models.rastrigin import Rastrigin
from src.models.queens import NQueens

m = Rastrigin()
# Rastrigin grid search
PARAMS_RASTRIGIN = {
    'num_generations': [50, 75, 100],
    'mutation_probability': [0.01, 0.05, 0.1],
    'pop_size': [10, 30, 60, 100],
    'crossover_probability': [0.6, 0.7, 0.8, 0.9]
}


# NQueens grid search
