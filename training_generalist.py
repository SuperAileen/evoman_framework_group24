import sys
import os
import time
import numpy as np
from math import fabs, sqrt
import glob
import sys

from evoman.environment import Environment
from demo_controller import player_controller

from deap import base, creator, tools, algorithms

import datetime

class GeneralistOptimizer:
    def __init__(self, base_experiment_name, enemy_set, n_hidden_neurons=10, n_population=100, n_generations=30,
                 mutation_rate=0.2, sigma=0.1, mode="GA"):

        enemy_string = '_'.join(map(str, enemy_set))
        parent_directory = f'experiments_train_generalist_{enemy_string}'

        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_name = os.path.join(parent_directory, f"{base_experiment_name}_generalist_{mode}_{current_time}")

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.enemy_set = enemy_set
        self.n_hidden_neurons = n_hidden_neurons
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.mode = mode

        self.env = self.setup_environment()
        self.n_vars = self.calculate_num_weights()

        self.toolbox = base.Toolbox()

        if self.mode == "GA":
            self.setup_ga()
        elif self.mode == "ES":
            self.setup_es()
        else:
            print("Hey! You can only choose GA or ES!")
            return

    def setup_environment(self):
        env = Environment(experiment_name=self.experiment_name,
                          enemies=self.enemy_set,
                          multiplemode="yes",
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)
        env.state_to_log()
        return env

    def calculate_num_weights(self):
        # Assuming the environment has a consistent sensor configuration
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
        # 修改重组策略为 Uniform Crossover
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        # 修改突变策略为随机重置突变，提升多样性
        self.toolbox.register("mutate", tools.mutUniformInt, low=dom_l, up=dom_u, indpb=self.mutation_rate)
        # # 修改选择策略为 Roulette Selection
        # self.toolbox.register("select", tools.selRoulette)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def setup_es(self):
        dom_u = 1
        dom_l = -1

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.toolbox.register("attr_float", np.random.uniform, dom_l, dom_u)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float,
                              n=self.n_vars)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate)
        # 使用适应性步长控制的突变函数
        self.toolbox.register("mutate", self.mutate_es_adaptive)
        self.toolbox.register("select", tools.selBest)

    def mutate_es_adaptive(self, individual):
        # 初始化步长
        if not hasattr(individual, 'sigma'):
            individual.sigma = np.ones(len(individual)) * self.sigma

        # 调整步长：使用1/5成功率规则调整
        tau = 1 / np.sqrt(2 * np.sqrt(len(individual)))
        tau_prime = 1 / np.sqrt(2 * len(individual))

        # 为每个维度添加随机噪声来调整步长
        individual.sigma *= np.exp(tau_prime * np.random.normal() + tau * np.random.normal(size=len(individual)))

        # 应用变异
        noise = np.random.normal(0, individual.sigma, len(individual))
        individual += noise

        # 确保个体的基因值在允许的范围内
        np.clip(individual, -1, 1, out=individual)

        return individual,

    def evaluate(self, individual):
        fitness = self.env.play(pcont=np.array(individual))[0]
        return fitness,

    def mutate_es(self, individual):
        noise = np.random.normal(0, self.sigma, len(individual))
        individual += noise
        return individual,

    def run(self):
        if self.mode == "GA":
            return self.run_ga()
        elif self.mode == "ES":
            return self.run_es()

        print("Invalid mode encountered during run.")

    def run_ga(self):
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

        best = hof[0]
        np.savetxt(f'{self.experiment_name}/best.txt', best)

        return population, logbook, hof

    def run_es(self):
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

        record = stats.compile(population)
        logbook.record(gen=0, **record)

        for gen in range(1, self.n_generations + 1):
            parents = list(map(self.toolbox.clone, population))
            offspring = parents
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                mutant.fitness.values = self.evaluate(mutant)

            offspring = offspring + parents

            population[:] = self.toolbox.select(offspring, len(population))

            hof.update(population)
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            print(f"Gen {gen}: {record}")

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

        fim = time.time()
        execution_time_minutes = round((fim - ini) / 60)
        execution_time_seconds = round((fim - ini))

        print(f'\nExecution time: {execution_time_minutes} minutes')
        print(f'Execution time: {execution_time_seconds} seconds')

        self.save_results()
        full_path = os.path.abspath(stats_file_path)
        print(full_path)
        return full_path

if __name__ == "__main__":
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_generalist'
    n_hidden_neurons = 10
    n_population = 100
    n_generations = 30
    mutation_rate = 0.2
    sigma = 0.1

    enemy_sets = [[1, 3, 4, 6], [2, 5, 7, 8]]
    modes = ["GA", "ES"]

    for enemy_set in enemy_sets:
        for mode in modes:
            generalist_optimizer = GeneralistOptimizer(experiment_name, enemy_set, n_hidden_neurons, n_population, n_generations,
                                                       mutation_rate, sigma, mode=mode)
            generalist_optimizer.execute()



            