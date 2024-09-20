import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re

def run_optimizer():
    cwd = os.getcwd()
    result = subprocess.run(['python', 'training_specialist_method_2_aileen.py'], cwd=cwd, capture_output=True, text=True)
    
    # Use regex to find the path to the stats.txt file
    print("Output from script:", result.stdout)  # Print the output from the script
    output_lines = result.stdout.strip().split('\n')
    stats_path = output_lines[-1]  # stats.txt path is the last line of the output
    if os.path.exists(stats_path):
        return stats_path
    else:
        raise Exception("Could not find stats.txt path in the output")

def aggregate_stats(runs):
    all_data = []
    for _ in range(runs):
        stats_path = run_optimizer()
        data = np.genfromtxt(stats_path, skip_header=1)
        all_data.append(data)
    
    all_data = np.array(all_data)
    mean_data = np.mean(all_data, axis=0)
    std_data = np.std(all_data, axis=0)
    
    aggregated_data = np.column_stack((mean_data[:,0], mean_data[:,1], mean_data[:,2], std_data[:,1], std_data[:,2]))
    return aggregated_data


def plot_aggregated_stats(aggregated_data, num_runs):
    generations = aggregated_data[:, 0]
    avg_fitness = aggregated_data[:, 1]
    max_fitness = aggregated_data[:, 2]
    std_dev = aggregated_data[:, 3]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, avg_fitness, label='Average Fitness', color='b')
    plt.fill_between(generations, avg_fitness - std_dev, avg_fitness + std_dev, color='b', alpha=0.2)
    plt.plot(generations, max_fitness, label='Max Fitness', color='r')
    plt.fill_between(generations, max_fitness - std_dev, max_fitness + std_dev, color='r', alpha=0.2)
    plt.title('Aggregated Fitness over Generations ({} runs) Using Method 2'.format(num_runs))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max(generations))
    plt.savefig('aggregated_fitness_plot.png')
    plt.show()


if __name__ == "__main__":
    num_runs = 3  # This should match with the number of runs in aggregate_stats
    aggregated_data = aggregate_stats(num_runs)
    plot_aggregated_stats(aggregated_data, num_runs)

