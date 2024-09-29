import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re
from training_specialist import EvolutAlgorithmOptimizer  # 导入类
# from training_specialist_lingfeng import save_gain_results, collect_gain_results
import pandas as pd


def run_optimizer(mode, enemy):
    # Set headless mode for faster experiments
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_train'
    # enemies = [1, 2, 3, 4, 5, 6, 7, 8]

    # 原本默认
    n_hidden_neurons = 10
    n_population = 100
    n_generations = 30
    mutation_rate = 0.2
    sigma = 0.1

    # GA
    if mode == "GA":
        n_hidden_neurons = 7
        n_population = 204
        n_generations = 40
        mutation_rate = 0.2883660095952316
        sigma = 0.1
    elif mode == "ES":
        n_hidden_neurons = 14
        n_population = 293
        n_generations = 40
        mutation_rate = 1
        sigma = 0.15382040354562315

    optimizer = EvolutAlgorithmOptimizer(experiment_name, [enemy], n_hidden_neurons, n_population, n_generations,
                                         mutation_rate, sigma, mode=mode)
    stats_path = optimizer.execute()

    if os.path.exists(stats_path):
        return stats_path
    else:
        raise Exception(f"Could not find stats.txt path for {mode}")


def aggregate_stats(mode, enemy, runs):
    all_data = []
    for _ in range(runs):
        stats_path = run_optimizer(mode, enemy)
        data = np.genfromtxt(stats_path, skip_header=1)
        all_data.append(data)

    all_data = np.array(all_data)
    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)

    aggregated_data = np.column_stack(
        (mean_data[:, 0], mean_data[:, 1], mean_data[:, 2], std_data[:, 1], std_data[:, 2]))
    return aggregated_data


def plot_aggregated_stats(aggregated_data_1, aggregated_data_2, enemy, num_runs):
    generations_1 = aggregated_data_1[:, 0]
    avg_fitness_1 = aggregated_data_1[:, 1]
    max_fitness_1 = aggregated_data_1[:, 2]
    std_dev_1 = aggregated_data_1[:, 3]

    generations_2 = aggregated_data_2[:, 0]
    avg_fitness_2 = aggregated_data_2[:, 1]
    max_fitness_2 = aggregated_data_2[:, 2]
    std_dev_2 = aggregated_data_2[:, 3]

    plt.figure(figsize=(10, 6))

    # Plot EA1 (GA)
    plt.plot(generations_1, avg_fitness_1, label='GA - Average Fitness', color='red', linestyle='--')
    plt.fill_between(generations_1, avg_fitness_1 - std_dev_1, avg_fitness_1 + std_dev_1, color='red', alpha=0.2)
    plt.plot(generations_1, max_fitness_1, label='GA - Max Fitness', color='red')
    plt.fill_between(generations_1, max_fitness_1 - std_dev_1, max_fitness_1 + std_dev_1, color='red', alpha=0.2)

    # Plot EA2 (ES)
    plt.plot(generations_2, avg_fitness_2, label='ES - Average Fitness', color='blue', linestyle='--')
    plt.fill_between(generations_2, avg_fitness_2 - std_dev_2, avg_fitness_2 + std_dev_2, color='blue', alpha=0.2)
    plt.plot(generations_2, max_fitness_2, label='ES - Max Fitness', color='blue')
    plt.fill_between(generations_2, max_fitness_2 - std_dev_2, max_fitness_2 + std_dev_2, color='blue', alpha=0.2)

    plt.title(f'Aggregated Fitness over Generations for GA and ES - Enemy {enemy} ({num_runs} runs)'.format(num_runs))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(False)
    plt.xlim(min(generations_1.min(), generations_2.min()) - 1, max(generations_1.max(), generations_2.max()) + 1)
    plt.savefig(f'Aggregated Fitness over Generations for GA and ES - Enemy {enemy} ({num_runs} runs).png')
    plt.show()


# if __name__ == "__main__":
#     num_runs = 10
#     enemy = 2
#
#     aggregated_data_1 = aggregate_stats("GA", enemy, num_runs)
#
#     aggregated_data_2 = aggregate_stats("ES", enemy, num_runs)
#
#     plot_aggregated_stats(aggregated_data_1, aggregated_data_2, enemy, num_runs)

if __name__ == "__main__":
    num_runs = 10

    for enemy in range(1, 9):
        aggregated_data_1 = aggregate_stats("GA", enemy, num_runs)
        aggregated_data_2 = aggregate_stats("ES", enemy, num_runs)

        plot_aggregated_stats(aggregated_data_1, aggregated_data_2, enemy, num_runs)
