import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re

def run_optimizer(script_name):
    cwd = os.getcwd()
    result = subprocess.run(['python', script_name], cwd=cwd, capture_output=True, text=True)
    
    # Use regex to find the path to the stats.txt file
    print(f"Output from {script_name}:", result.stdout)  # Print the output from the script
    output_lines = result.stdout.strip().split('\n')
    stats_path = output_lines[-1]  # stats.txt path is the last line of the output
    if os.path.exists(stats_path):
        return stats_path
    else:
        raise Exception(f"Could not find stats.txt path in the output from {script_name}")

def aggregate_stats(script_name, runs):
    all_data = []
    for _ in range(runs):
        stats_path = run_optimizer(script_name)
        data = np.genfromtxt(stats_path, skip_header=1)
        all_data.append(data)
    
    all_data = np.array(all_data)
    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)
    
    aggregated_data = np.column_stack((mean_data[:,0], mean_data[:,1], mean_data[:,2], std_data[:,1], std_data[:,2]))
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

    plt.figure(figsize=(10, 5))
    
    # Plot method 1
    plt.plot(generations_1, avg_fitness_1, label='Method 1 - Average Fitness', color='b')
    plt.fill_between(generations_1, avg_fitness_1 - std_dev_1, avg_fitness_1 + std_dev_1, color='b', alpha=0.2)
    plt.plot(generations_1, max_fitness_1, label='Method 1 - Max Fitness', color='r')
    plt.fill_between(generations_1, max_fitness_1 - std_dev_1, max_fitness_1 + std_dev_1, color='r', alpha=0.2)

    # Plot method 2
    plt.plot(generations_2, avg_fitness_2, label='Method 2 - Average Fitness', color='g')
    plt.fill_between(generations_2, avg_fitness_2 - std_dev_2, avg_fitness_2 + std_dev_2, color='g', alpha=0.2)
    plt.plot(generations_2, max_fitness_2, label='Method 2 - Max Fitness', color='orange')
    plt.fill_between(generations_2, max_fitness_2 - std_dev_2, max_fitness_2 + std_dev_2, color='orange', alpha=0.2)

    plt.title('Aggregated Fitness over Generations for Method 1 and Method 2 ({} runs)'.format(num_runs))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.xlim(min(generations_1.min(), generations_2.min()) - 1, max(generations_1.max(), generations_2.max()) + 1)
    plt.savefig('aggregated_fitness_comparison.png')
    plt.show()


if __name__ == "__main__":
    num_runs = 10 
    
    # Aggregate data for method 1
    aggregated_data_1 = aggregate_stats('training_specialist_method_1_jiawei.py', num_runs)
    
    # Aggregate data for method 2
    aggregated_data_2 = aggregate_stats('training_specialist_method_2_aileen.py', num_runs)
    
    # Plot both methods on the same graph
    plot_aggregated_stats(aggregated_data_1, aggregated_data_2, num_runs)
