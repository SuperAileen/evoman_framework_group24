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

import datetime
from line_plot_once_method_1_jiawei import plot_stats

class EvolutAlgorithmOptimizer:
    def __init__(self, base_experiment_name, enemies, n_hidden_neurons=10, n_population=100, n_generations=30,
                 mutation_rate=0.2, sigma=0.1, mode = "GA"):

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_name = f"{base_experiment_name}_{enemies}_{mode}_{current_time}"

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.enemies = enemies
        self.n_hidden_neurons = n_hidden_neurons
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.mode = mode

        self.env = self.setup_environment()
        self.n_vars = self.calculate_num_weights()

        self.toolbox = base.Toolbox()

        # setup for GA or ES
        if self.mode == "GA":
            self.setup_ga()
        elif self.mode == "ES":
            self.setup_es()
        else:
            print("Hey! You can only choose GA or ESsssssssssssssssssssssssssssssssssssssssssssssssss!")
            return

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
        return (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

    def setup_ga(self):
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

    def setup_es(self):
        dom_u = 1
        dom_l = -1

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.toolbox.register("attr_float", np.random.uniform, dom_l, dom_u)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.n_vars)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mutate", self.mutate_es)
        self.toolbox.register("select", tools.selBest)

    def evaluate(self, individual):
        return self.env.play(pcont=np.array(individual))[0],  # Return the fitness

    def mutate_es(self, individual):
        # Apply Gaussian mutation with sigma as standard deviation
        noise = np.random.normal(0, self.sigma, len(individual))
        individual += noise
        return individual,

    def run(self):
        if self.mode == "GA":
            return self.run_GA()
        elif self.mode == "ES":
            return self.run_ES()

        print("Hey!!!!!! You should not meet this printttttttttttttttttttttttttttttt!!!!!!")

    def run_GA(self):
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

        return population, logbook, hof

    def run_ES(self):
        # Create initial population
        population = self.toolbox.population(n=self.n_population)

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Initialize logbook to store statistics
        logbook = tools.Logbook()
        logbook.header = ['gen', 'avg', 'max', 'std']  # Customize the header if needed

        # Hall of Fame setup
        hof = tools.HallOfFame(1, similar=np.array_equal)

        # DEAP Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Evolutionary loop
        for gen in range(self.n_generations):
            parents = list(map(self.toolbox.clone, population))  # Clone population for mutation
            offspring = parents

            # Mutate and evaluate offspring
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                # mutant.fitness.values = self.toolbox.evaluate(mutant)
                mutant.fitness.values = self.evaluate(mutant)

            offspring = offspring + parents

            # Select the best individuals to survive
            population[:] = self.toolbox.select(offspring, len(population))

            # Update Hall of Fame with the best individuals
            hof.update(population)

            # Record statistics and add to logbook
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            print(f"Gen {gen}: {record}")

        # Save the best individual to a file
        best = hof[0]
        np.savetxt(f'{self.experiment_name}/best.txt', best)

        # Return population, logbook, and Hall of Fame
        return population, logbook, hof

    def save_results(self):
        file = open(f'{self.experiment_name}/neuroended', 'w')
        file.close()
        self.env.state_to_log()

    def execute(self):
        ini = time.time()

        if self.mode == "GA":
            print('Starting Genetic Algorithm Optimization...')
        elif self.mode == "ES":
            print('Starting Evolution Strategies Optimization...')
        population, logbook, hof = self.run()

        # Process and save detailed statistics from the logbook
        gen = logbook.select("gen")
        avg = logbook.select("avg")
        max_ = logbook.select("max")
        std = logbook.select("std")

        stats_data = np.column_stack((gen, avg, max_, std))  # Combine the statistics into a single array
        stats_file_path = f'{self.experiment_name}/stats.txt'
        np.savetxt(stats_file_path, stats_data, header='gen avg max std', comments='', fmt='%f')

        # Call the plot_stats function to generate the plot
        plot_stats(stats_file_path, self.n_generations, self.experiment_name, f"mode:{self.mode}, enemy:{self.enemies}")

        fim = time.time()
        execution_time_minutes = round((fim - ini) / 60)
        execution_time_seconds = round((fim - ini))

        print(f'\nExecution time: {execution_time_minutes} minutes')
        print(f'Execution time: {execution_time_seconds} seconds')

        self.save_results()
        full_path = os.path.abspath(stats_file_path)
        print(full_path)
        return full_path  # Return the path to the stats file for further analysis


if __name__ == "__main__":
    # Set headless mode for faster experiments
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Parameters
    experiment_name = 'optimization_test_method_1'
    enemies = [8]
    n_hidden_neurons = 10
    n_population = 100
    n_generations = 30
    mutation_rate = 0.2

    # Initialize and execute the optimizer
    optimizer1 = EvolutAlgorithmOptimizer(experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
                                          mutation_rate, sigma=0.1, mode = "GA")
    optimizer1.execute()

    optimizer2 = EvolutAlgorithmOptimizer(experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
                                          mutation_rate, sigma=0.1, mode = "ES")
    optimizer2.execute()