import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_stats(file_path, n_generations, output_dir):
    data = np.genfromtxt(file_path, skip_header=1)
    generations = data[:, 0]
    avg_fitness = data[:, 1]
    max_fitness = data[:, 2]
    std_dev = data[:, 3]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, avg_fitness, label='Average Fitness', color='b')
    plt.fill_between(generations, avg_fitness - std_dev, avg_fitness + std_dev, color='b', alpha=0.2)
    plt.plot(generations, max_fitness, label='Max Fitness', color='r')
    plt.fill_between(generations, max_fitness - std_dev, max_fitness + std_dev, color='r', alpha=0.2)
    plt.title('Fitness over Generations Using Method 2')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.xlim(generations.min() - 1, generations.max() + 1)
    plt.savefig(os.path.join(output_dir, 'fitness_plot.png'))  # Save the plot in the specified directory
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        output_dir = sys.argv[2]
        n_generations = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        plot_stats(file_path, n_generations, output_dir)
    else:
        print("Usage: python line_plot_once_method_2_aileen.py <path_to_stats_file> <output_directory> <number_of_generations>")
