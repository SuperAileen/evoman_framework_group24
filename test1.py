###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

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

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Genetic Algorithm    ###

ini = time.time()  # sets time marker

# genetic algorithm params
run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation_rate = 0.2

# Define fitness function
def evaluate(individual):
    return env.play(pcont=np.array(individual))[0],

# Create types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", np.random.uniform, dom_l, dom_u)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operator registering
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=mutation_rate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create initial population
population = toolbox.population(n=npop)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# DEAP Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Define the hall of fame
hof = tools.HallOfFame(1, similar=np.array_equal)

# Run the algorithm
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=mutation_rate, 
                                          ngen=gens, stats=stats, halloffame=hof, verbose=True)

# Get best solution
best = hof[0]

# Save best solution
np.savetxt(experiment_name+'/best.txt', best)

fim = time.time() # prints total execution time for experiment
print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print('\nExecution time: '+str(round((fim-ini)))+' seconds \n')

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state
