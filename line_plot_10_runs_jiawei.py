import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re


def run_optimizer(script_name, mode):
    cwd = os.getcwd()
    result = subprocess.run([r'C:\Users\liaojw\Documents\study\course\Evolutionary Computing\assignment1\evoman_framework_group24\venv\Scripts\python.exe', script_name, mode], cwd=cwd, capture_output=True, text=True)

    print(f"Output from {script_name}:", result.stdout)
    output_lines = result.stdout.strip().split('\n')
    stats_path = output_lines[-1]
    if os.path.exists(stats_path):
        return stats_path
    else:
        raise Exception(f"Could not find stats.txt path in the output from {script_name}")


def aggregate_stats(script_name, mode, runs):
    all_data = []
    for _ in range(runs):
        stats_path = run_optimizer(script_name, mode)
        data = np.genfromtxt(stats_path, skip_header=1)
        all_data.append(data)

    all_data = np.array(all_data)
    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)

    aggregated_data = np.column_stack(
        (mean_data[:, 0], mean_data[:, 1], mean_data[:, 2], std_data[:, 1], std_data[:, 2]))
    return aggregated_data


def plot_aggregated_stats(aggregated_data_1, aggregated_data_2, num_runs):
    generations_1 = aggregated_data_1[:, 0]
    avg_fitness_1 = aggregated_data_1[:, 1]
    max_fitness_1 = aggregated_data_1[:, 2]
    std_dev_1 = aggregated_data_1[:, 3]

    generations_2 = aggregated_data_2[:, 0]
    avg_fitness_2 = aggregated_data_2[:, 1]
    max_fitness_2 = aggregated_data_2[:, 2]
    std_dev_2 = aggregated_data_2[:, 3]

    plt.figure(figsize=(10, 6))

    # Plot EA1
    plt.plot(generations_1, avg_fitness_1, label='GA - Average Fitness', color='red', linestyle='--')
    plt.fill_between(generations_1, avg_fitness_1 - std_dev_1, avg_fitness_1 + std_dev_1, color='red', alpha=0.2)
    plt.plot(generations_1, max_fitness_1, label='GA - Max Fitness', color='red')
    plt.fill_between(generations_1, max_fitness_1 - std_dev_1, max_fitness_1 + std_dev_1, color='red', alpha=0.2)

    # Plot EA2
    plt.plot(generations_2, avg_fitness_2, label='ES - Average Fitness', color='blue', linestyle='--')
    plt.fill_between(generations_2, avg_fitness_2 - std_dev_2, avg_fitness_2 + std_dev_2, color='blue', alpha=0.2)
    plt.plot(generations_2, max_fitness_2, label='ES - Max Fitness', color='blue')
    plt.fill_between(generations_2, max_fitness_2 - std_dev_2, max_fitness_2 + std_dev_2, color='blue', alpha=0.2)

    plt.title('Aggregated Fitness over Generations for GA and ES - Enemy 2 (Air Man) ({} runs)'.format(num_runs))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(False)
    plt.xlim(min(generations_1.min(), generations_2.min()) - 1, max(generations_1.max(), generations_2.max()) + 1)
    plt.savefig('aggregated_fitness_comparison_enemy_2_air_man.png')
    plt.show()


if __name__ == "__main__":
    num_runs = 10

    aggregated_data_1 = aggregate_stats('training_specialist_jiawei.py', "GA", num_runs)

    aggregated_data_2 = aggregate_stats('training_specialist_jiawei.py', "ES", num_runs)

    plot_aggregated_stats(aggregated_data_1, aggregated_data_2, num_runs)
