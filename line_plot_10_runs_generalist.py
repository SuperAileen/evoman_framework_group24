import numpy as np
import matplotlib.pyplot as plt
import os
from training_generalist import GeneralistOptimizer  # Updated import
import pandas as pd


def run_optimizer(mode, enemy_set):
    # Set headless mode for faster experiments
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_generalist'

    # Parameters from the generalist file
    n_hidden_neurons = 10
    n_population = 100
    n_generations = 30
    mutation_rate = 0.2
    sigma = 0.1
    k = 5

    optimizer = GeneralistOptimizer(experiment_name, enemy_set, n_hidden_neurons, n_population, n_generations,
                                    mutation_rate, sigma, mode=mode, k = k)
    stats_path = optimizer.execute()

    if os.path.exists(stats_path):
        return stats_path
    else:
        raise Exception(f"Could not find stats.txt path for {mode}")


def aggregate_stats(mode, enemy_set, runs):
    all_data = []
    for _ in range(runs):
        stats_path = run_optimizer(mode, enemy_set)
        data = np.genfromtxt(stats_path, skip_header=1)
        all_data.append(data)

    all_data = np.array(all_data)
    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)

    aggregated_data = np.column_stack(
        (mean_data[:, 0], mean_data[:, 1], mean_data[:, 2], std_data[:, 1], std_data[:, 2]))
    return aggregated_data


def plot_aggregated_stats(aggregated_data_1, aggregated_data_2, enemy_set, num_runs):
    generations_1 = aggregated_data_1[:, 0]
    avg_fitness_1 = aggregated_data_1[:, 1]
    max_fitness_1 = aggregated_data_1[:, 2]
    std_dev_1 = aggregated_data_1[:, 3]

    generations_2 = aggregated_data_2[:, 0]
    avg_fitness_2 = aggregated_data_2[:, 1]
    max_fitness_2 = aggregated_data_2[:, 2]
    std_dev_2 = aggregated_data_2[:, 3]

    plt.figure(figsize=(10, 6))

    plt.plot(generations_1, avg_fitness_1, label='Diversity-oriented GA - Average Fitness', color='red', linestyle='--')
    plt.fill_between(generations_1, avg_fitness_1 - std_dev_1, avg_fitness_1 + std_dev_1, color='red', alpha=0.2)
    plt.plot(generations_1, max_fitness_1, label='Diversity-oriented GA - Max Fitness', color='red')
    plt.fill_between(generations_1, max_fitness_1 - std_dev_1, max_fitness_1 + std_dev_1, color='red', alpha=0.2)

    plt.plot(generations_2, avg_fitness_2, label='Elitism-oriented GA - Average Fitness', color='blue', linestyle='--')
    plt.fill_between(generations_2, avg_fitness_2 - std_dev_2, avg_fitness_2 + std_dev_2, color='blue', alpha=0.2)
    plt.plot(generations_2, max_fitness_2, label='Elitism-oriented GA - Max Fitness', color='blue')
    plt.fill_between(generations_2, max_fitness_2 - std_dev_2, max_fitness_2 + std_dev_2, color='blue', alpha=0.2)

    plt.title(f'Fitness over Generations - Enemies {enemy_set} ({num_runs} runs)', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)

    plt.ylim(0, 100)

    plt.legend(prop={'size': 12})

    plt.grid(False)
    plt.xlim(min(generations_1.min(), generations_2.min()) - 1, max(generations_1.max(), generations_2.max()) + 1)

    plt.savefig(f'Aggregated_Fitness_GA_ES_Enemies_{enemy_set}_{num_runs}_runs.png')
    plt.show()


if __name__ == "__main__":
    num_runs = 10

    enemy_sets = [[1, 3, 4, 6], [2, 5, 7, 8]]

    for enemy_set in enemy_sets:
        aggregated_data_1 = aggregate_stats("GA1", enemy_set, num_runs)
        aggregated_data_2 = aggregate_stats("GA2", enemy_set, num_runs)

        plot_aggregated_stats(aggregated_data_1, aggregated_data_2, enemy_set, num_runs)



