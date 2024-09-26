import sys
import os
import time
import numpy as np
from math import fabs, sqrt
import glob
import sys
import pandas as pd 

from evoman.environment import Environment
from demo_controller import player_controller

# imports DEAP library
from deap import base, creator, tools, algorithms

import datetime
# from line_plot_once_method_1_jiawei import plot_stats

class EvolutAlgorithmOptimizer:
    def __init__(self, base_experiment_name, enemies, n_hidden_neurons=10, n_population=100, n_generations=30,
                 mutation_rate=0.2, sigma=0.1, mode = "GA"):
        
        parent_directory = 'experiments_train_lingfeng'
        parent_directory = f'{parent_directory}_{enemies[0]}'  # make sure the parent directory is unique

        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_name = os.path.join(parent_directory, f"{base_experiment_name}_{enemies}_{mode}_{current_time}")

        #self.experiment_name = os.path.join(parent_directory, f"{base_experiment_name}_{current_time}")

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

        self.gain_record=[]

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

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.toolbox.register("attr_float", np.random.uniform, dom_l, dom_u)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float,
                              n=self.n_vars)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("gain_f", self.gain_f)
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
        self.toolbox.register("gain_f", self.gain_f)
        self.toolbox.register("mutate", self.mutate_es)
        self.toolbox.register("select", tools.selBest)

    def evaluate(self, individual):
        return self.env.play(pcont=np.array(individual))[0],  

    def gain_f(self, individual):

        result=self.env.play(pcont=np.array(individual))
        player_life = result[1]
        enemy_life = result[2]
        individual_gain = player_life - enemy_life
        return individual_gain, 

    def gain_frame(self, avg_gain, max_gain):   
        record = {
            'enemies': self.enemies,
            'n_hidden_neurons': self.n_hidden_neurons,
            'n_population': self.n_population,
            'n_generations': self.n_generations,
            'mutation_rate': self.mutation_rate,
            'sigma': self.sigma,
            'mode': self.mode,
            'avg_gain': avg_gain,
            'max_gain': max_gain
        }
        self.gain_record.append(record)

    def mutate_es(self, individual):
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
        population = self.toolbox.population(n=self.n_population)

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1, similar=np.array_equal)

        population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=self.mutation_rate,
                                                  ngen=self.n_generations, stats=stats, halloffame=hof, verbose=True)
        
        # final individual gain
        gainness = list(map(self.toolbox.gain_f, population))   
        for ind, (gain, ) in zip(population, gainness):
            ind.gain = gain

        avg_gain = np.mean([ind.gain for ind in population])
        max_gain = np.max([ind.gain for ind in population])
        print(f'Average gain: {avg_gain}, max gain: {max_gain} in the last generation')

        self.gain_frame(avg_gain, max_gain)

        best = hof[0]
        np.savetxt(f'{self.experiment_name}/best.txt', best)

        return population, logbook, hof

    def run_ES(self):
        population = self.toolbox.population(n=self.n_population)

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        logbook = tools.Logbook()
        logbook.header = ['gen', 'avg', 'max', 'std']

        hof = tools.HallOfFame(1, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        for gen in range(self.n_generations+1):
            parents = list(map(self.toolbox.clone, population)) 
            offspring = parents
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                # mutant.fitness.values = self.toolbox.evaluate(mutant)
                mutant.fitness.values = self.evaluate(mutant)

            offspring = offspring + parents

            population[:] = self.toolbox.select(offspring, len(population))

            hof.update(population)
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            print(f"Gen {gen}: {record}")

        gainness = list(map(self.toolbox.gain_f, population))
        for ind, (gain, ) in zip(population, gainness):
            ind.gain = gain

        avg_gain = np.mean([ind.gain for ind in population])
        max_gain = np.max([ind.gain for ind in population])
        print(f'Average gain: {avg_gain}, Max gain: {max_gain} in the last generation')

        self.gain_frame(avg_gain, max_gain)

        best = hof[0]
        np.savetxt(f'{self.experiment_name}/best.txt', best)

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

        gen = logbook.select("gen")
        avg = logbook.select("avg")
        max_ = logbook.select("max")
        std = logbook.select("std")

        stats_data = np.column_stack((gen, avg, max_, std)) 
        stats_file_path = f'{self.experiment_name}/stats.txt'
        np.savetxt(stats_file_path, stats_data, header='gen avg max std', comments='', fmt='%f')

        # plot_stats(stats_file_path, self.n_generations, self.experiment_name, f"mode:{self.mode}, enemy:{self.enemies}")

        fim = time.time()
        execution_time_minutes = round((fim - ini) / 60)
        execution_time_seconds = round((fim - ini))

        print(f'\nExecution time: {execution_time_minutes} minutes')
        print(f'Execution time: {execution_time_seconds} seconds')

        self.save_results()
        full_path = os.path.abspath(stats_file_path)
        print(full_path)
    

        return full_path 
    
    def get_gain_records(self):
        
        return pd.DataFrame(self.gain_record)
    
def collect_gain_results(optimizers):

    all_gain_data = pd.DataFrame(columns=[
        'enemies', 'n_hidden_neurons', 'n_population', 'n_generations', 'mutation_rate', 'sigma', 'mode', 'avg_gain', 'max_gain'
    ])

    for optimizer in optimizers:
        gain_data = optimizer.get_gain_records()  # 获取每个 optimizer 的 gain 记录
        all_gain_data = pd.concat([all_gain_data, gain_data], ignore_index=True)  # 将结果追加到总的 DataFrame

    return all_gain_data


def save_gain_results(all_gain_data, experiment_name):
    
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    file_path = f'{experiment_name}/all_gain_results_{np.random.randint(0, 9999)}.csv'
    all_gain_data.to_csv(file_path, index=False)
    print(f'All gain results saved to {file_path}')




if __name__ == "__main__":
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_train_lingfeng'
    gain_path = 'gain_dataframe'
    n_hidden_neurons = 10
    n_population = 100
    n_generations = 30
    mutation_rate = 0.2
    optimizer_list = []

    enemies_list=[[i] for i in range(1,9)]

    for enemies in enemies_list:

        print(f"Running optimization for enemies: {enemies}") 

        if len(sys.argv) == 1:
            optimizer1 = EvolutAlgorithmOptimizer(experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
                                                mutation_rate, sigma=0.1, mode = "GA")
            optimizer1.execute()
            optimizer_list.append(optimizer1)

            optimizer2 = EvolutAlgorithmOptimizer(experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
                                                mutation_rate, sigma=0.1, mode = "ES")
            optimizer2.execute()
            optimizer_list.append(optimizer2)
        elif len(sys.argv) > 1:
            mode = sys.argv[1]
            optimizer = EvolutAlgorithmOptimizer(experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
                                                mutation_rate, sigma=0.1, mode=mode)
            optimizer.execute()
            optimizer_list.append(optimizer)


    all_gain_data = collect_gain_results(optimizer_list)
    print(all_gain_data)

    save_gain_results(all_gain_data=all_gain_data, experiment_name=experiment_name)