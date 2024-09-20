import numpy as np
import matplotlib.pyplot as plt

def plot_stats(file_path, n_generations):
    data = np.genfromtxt(file_path, skip_header=1)  # Using genfromtxt for better handling of irregular formats
    generations = data[:, 0]
    avg_fitness = data[:, 1]
    max_fitness = data[:, 2]
    std_dev = data[:, 3]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, avg_fitness, label='Average Fitness', color='b')
    plt.fill_between(generations, avg_fitness - std_dev, avg_fitness + std_dev, color='b', alpha=0.2)
    plt.plot(generations, max_fitness, label='Max Fitness', color='r')
    plt.fill_between(generations, max_fitness - std_dev, max_fitness + std_dev, color='r', alpha=0.2)
    plt.title('Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, n_generations)  # Set x-axis limit to the specified number of generations
    plt.savefig('fitness_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_stats('path_to_your_stats_file/stats.txt', 30)
