from skopt import Optimizer
from skopt.space import Real, Integer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re
from training_specialist_lingfeng import EvolutAlgorithmOptimizer  # 导入类 
from training_specialist_lingfeng import save_gain_results, collect_gain_results


def bayesian_optimization(experiment_name, enemies, mode="GA", n_calls=20):
    # 根据模式定义不同的超参数搜索空间
    if mode == "GA":
        search_space = [
            Integer(5, 20, name='n_hidden_neurons'),  # 隐藏层神经元数
            Integer(50, 300, name='n_population'),    # 种群大小
            Integer(20, 50, name='n_generations'),    # 世代数
            Real(0.1, 0.3, name='mutation_rate')      # 突变率
            # GA模式下不包含sigma，直接设定为0.1
        ]
    elif mode == "ES":
        search_space = [
            Integer(5, 20, name='n_hidden_neurons'),  # 隐藏层神经元数
            Integer(50, 300, name='n_population'),    # 种群大小
            Integer(20, 50, name='n_generations'),    # 世代数
            Real(0.1, 0.3, name='mutation_rate'),     # 突变率
            Real(0.05, 0.2, name='sigma')             # Sigma, ES模式包含sigma
        ]

    # 初始化贝叶斯优化器
    optimizer = Optimizer(search_space)

    best_score = -float('inf')
    best_params = None
    all_results = []

    for i in range(n_calls):
        # 获取下一组超参数组合
        suggested_params = optimizer.ask()

        # 解包超参数组合
        if mode == "GA":
            n_hidden_neurons, n_population, n_generations, mutation_rate = suggested_params
            sigma = 0.1  # GA模式下，sigma设定为默认值0.1
        elif mode == "ES":
            n_hidden_neurons, n_population, n_generations, mutation_rate, sigma = suggested_params

        print(f"Running trial {i + 1}/{n_calls} for mode {mode}")
        print(f"Hyperparameters: hidden_neurons={n_hidden_neurons}, population={n_population}, "
              f"generations={n_generations}, mutation_rate={mutation_rate}, sigma={sigma if mode == 'ES' else '0.1'}")

        # 初始化优化器，使用指定的单个敌人
        optimizer_instance = EvolutAlgorithmOptimizer(
            experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
            mutation_rate, sigma, mode=mode  # 可以选择 GA 或 ES
        )
        optimizer_instance.execute()

        # 获取优化结果
        results = optimizer_instance.saveget_fitness()
        
        fitness = results['max_fitness'].item()  # 获取fitness作为评价指标

        print(f"Fitness for this trial: {fitness}")

        # 将当前的超参数和结果保存
        all_results.append({
            'n_hidden_neurons': n_hidden_neurons,
            'n_population': n_population,
            'n_generations': n_generations,
            'mutation_rate': mutation_rate,
            'sigma': sigma if mode == "ES" else 0.1,  # GA模式下sigma为0.1
            'fitness': fitness
        })

        # 更新贝叶斯优化器的结果
        optimizer.tell(suggested_params, -fitness)  # 使用负 fitness 因为 Optimizer 寻找最小化

        # 记录最佳超参数
        if fitness > best_score:
            best_score = fitness
            best_params = {
                'n_hidden_neurons': n_hidden_neurons,
                'n_population': n_population,
                'n_generations': n_generations,
                'mutation_rate': mutation_rate,
                'sigma': sigma if mode == "ES" else 0.1  # GA模式下sigma为0.1
            }

    print(f"Best Hyperparameters for mode {mode}: {best_params}")
    print(f"Best Fitness: {best_score}")

    output_dir = f'{experiment_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存所有结果为 CSV
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(f'{output_dir}/bayes_search_results_{mode}.csv', index=False)
    print(f"Bayesian optimization results saved to {output_dir}/bayes_search_results_{mode}.csv")

    return best_params, best_score


if __name__ == "__main__":
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_parameter'
    
    # 选择一个特定的敌人，例如敌人1
    enemies = [4]

    # 执行贝叶斯优化超参数搜索
    '''
    print("Running GA optimization:")
    best_params_ga, best_score_ga = bayesian_optimization(experiment_name, enemies, mode="GA", n_calls=20)
    print(f"Best Hyperparameters for GA: {best_params_ga}")
    '''

    print("\nRunning ES optimization:")
    best_params_es, best_score_es = bayesian_optimization(experiment_name, enemies, mode="ES", n_calls=20)
    print(f"Best Hyperparameters for ES: {best_params_es}")