# imports framework
import sys
import os
import time
import numpy as np
from math import fabs, sqrt
import glob

from evoman.environment import Environment
from demo_controller import player_controller

# imports DEAP library
from deap import base, creator, tools, algorithms


class GeneticAlgorithmOptimizer:
    def __init__(self, experiment_name, enemies, n_hidden_neurons=10, n_population=100, n_generations=30,
                 mutation_rate=0.2):
        self.experiment_name = experiment_name
        self.enemies = enemies
        self.n_hidden_neurons = n_hidden_neurons
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate

        # Environment setup
        self.env = self.setup_environment()
        self.n_vars = self.calculate_num_weights()

        # DEAP setup
        self.toolbox = base.Toolbox()
        self.setup_deap()

        # Create experiment directory
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

    def setup_environment(self):
        env = Environment(experiment_name=self.experiment_name,
                          enemies=self.enemies,
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)
        env.state_to_log()
        return env

    def calculate_num_weights(self):
        # Calculate the number of weights for multilayer with hidden neurons
        return (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

    def setup_deap(self):
        dom_u = 1
        dom_l = -1

        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        # Register attribute generator and individual population structure
        self.toolbox.register("attr_float", np.random.uniform, dom_l, dom_u)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float,
                              n=self.n_vars)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register operators
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=self.mutation_rate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual):
        return self.env.play(pcont=np.array(individual))[0],

    def run(self):
        # Create initial population
        population = self.toolbox.population(n=self.n_population)

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # DEAP Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of Fame setup
        hof = tools.HallOfFame(1, similar=np.array_equal)

        # Run the genetic algorithm
        population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=self.mutation_rate,
                                                  ngen=self.n_generations, stats=stats, halloffame=hof, verbose=True)

        # Get the best individual from the hall of fame
        best = hof[0]
        np.savetxt(f'{self.experiment_name}/best.txt', best)

        return best

    def save_results(self):
        file = open(f'{self.experiment_name}/neuroended', 'w')
        file.close()
        self.env.state_to_log()

    def execute(self):
        ini = time.time()

        print('Starting Genetic Algorithm Optimization...')
        best = self.run()

        fim = time.time()
        execution_time_minutes = round((fim - ini) / 60)
        execution_time_seconds = round((fim - ini))

        print(f'\nExecution time: {execution_time_minutes} minutes')
        print(f'Execution time: {execution_time_seconds} seconds')

        self.save_results()


if __name__ == "__main__":
    # Set headless mode for faster experiments
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Parameters
    experiment_name = 'optimization_test'
    enemies = [8]
    n_hidden_neurons = 10
    n_population = 100
    n_generations = 30
    mutation_rate = 0.2

    # Initialize and execute the optimizer
    optimizer = GeneticAlgorithmOptimizer(experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
                                          mutation_rate)
    optimizer.execute()
