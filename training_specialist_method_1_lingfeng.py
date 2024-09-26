# imports framework
import sys
import os
import time
import numpy as np
from math import fabs, sqrt
import glob
import random 

from evoman.environment import Environment
from demo_controller import player_controller

# imports DEAP library
from deap import base, creator, tools, algorithms

import datetime
from line_plot_once_method_1_jiawei import plot_stats

class GeneticAlgorithmOptimizer1:

    def __init__(self, base_experiment_name, enemies, n_hidden_neurons=10, n_population=100, n_generations=30,
                 mutation_rate=0.2):
        # Append current datetime to make the experiment name unique
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_name = f"{base_experiment_name}_{current_time}"

        parent_directory = 'experiments_train_lingfeng'
        #random_number = random.randint(1, 9999)  # Generate a random number between 1000 and 9999
        parent_directory = f'{parent_directory}_{enemies[0]}'

        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        # Create a unique experiment directory inside the parent directory
        self.experiment_name = os.path.join(parent_directory, f"{base_experiment_name}_{current_time}")

        # Create experiment directory before environment setup
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.enemies = enemies
        self.n_hidden_neurons = n_hidden_neurons
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate

        # Environment setup after ensuring directory is created
        self.env = self.setup_environment()
        self.n_vars = self.calculate_num_weights()

        # DEAP setup
        self.toolbox = base.Toolbox()
        self.setup_deap()

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
                              n=self.n_vars)  # Individual structure is an array of weights (floats)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)  # Population structure a list of individuals

        # Register operators
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("gain_f", self.gain_f)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=self.mutation_rate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    
    def init_individual_with_gain(self):
            ind = creator.Individual(np.random.rand(self.n_vars))
            ind.gain = 0.0  # Initialize the gain attribute with a default value (e.g., 0.0)
            return ind
    

    def evaluate(self, individual):
        # Assuming that the environment returns (fitness, player_life, enemy_life)
        result = self.env.play(pcont=np.array(individual))
        fitness = result[0]  # Get fitness


        # Return only the fitness, since we don't want to log the gain into the file
        return fitness,

    def gain_f(self, individual):

        result = self.env.play(pcont=np.array(individual))
        player_life = result[1]  # Player's remaining life
        enemy_life = result[2]  # Enemy's remaining life

        individual_gain = player_life - enemy_life  #

        return individual_gain,

        

    def run(self):
        # Create initial population
        #population = self.toolbox.population(n=self.n_population) # Generate the initial population of individuals， the weights of the neural network are initialized randomly
        population = [self.init_individual_with_gain() for _ in range(self.n_population)]
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, population))  # Evaluate all individuals in the population
        for ind, (fit, ) in zip(population, fitnesses): 
            ind.fitness.values = (fit,)  # Assign the fitness to the individual
        
        ## 问题还是出在没定义gain上面 creator 
        gainness=list(map(self.toolbox.gain_f, population))
        for ind, (gain, ) in zip(population, gainness):
            ind.gainness.values = gain 
        
    

        # DEAP Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)


        
            

        hof = tools.HallOfFame(1, similar=np.array_equal)


        # Run the genetic algorithm
        population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=self.mutation_rate,
                                                  ngen=self.n_generations, stats=stats, halloffame=hof, verbose=True)


       

        # Get the best individual from the hall of fame
        best = hof[0]
        np.savetxt(f'{self.experiment_name}/best.txt', best)

        return population, logbook, hof



    def save_results(self):
        file = open(f'{self.experiment_name}/neuroended', 'w')
        file.close()
        self.env.state_to_log()

    def execute(self):
        ini = time.time()

        print('Starting Genetic Algorithm Optimization...')
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
        plot_stats(stats_file_path, self.n_generations, self.experiment_name)

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
    experiment_name = 'optimization_train_lingfeng_m1'
    enemies_list = [[2], [3], [5]]  # List of enemy combinations
    n_hidden_neurons = 10
    n_population = 100
    n_generations = 30
    mutation_rate = 0.2

    # Iterate over each enemy combination and run the optimizer
    for enemies in enemies_list:
        print(f"\nStarting optimization for enemies: {enemies}")
        optimizer = GeneticAlgorithmOptimizer1(experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
                                               mutation_rate)
        optimizer.execute()






