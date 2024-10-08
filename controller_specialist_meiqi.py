#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
from evoman.environment import Environment
from demo_controller import player_controller

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

def find_best_solution(method, enemy):
    print(f'optimization_train_Jiawei_{enemy}_{method}_*'+'/best.txt')
    file_name = f'optimization_train_Jiawei_{enemy}_{method}_*'+'/best.txt'
    best_files = glob.glob(file_name)
    # best_files = glob.glob(f'optimization_test_[[]{enemy}[]]_{method}_*'+'/best.txt')
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",best_files)
    best_fitness = -np.inf
    best_solution = None

    for file in best_files:
        sol = np.loadtxt(file)
        fitness, _, _, _ = env.play(sol)
        
        if fitness > best_fitness:
            best_solution = sol

    return best_solution

n_hidden_neurons = 10

env = Environment(experiment_name='best_experiment_test',
                playermode="ai",
				player_controller=player_controller(n_hidden_neurons),
				speed="normal",
				enemymode="static",
                logs="off",
                randomini="yes",
				level=2,
				visuals=True)

results = {}
method_list = ['ES', 'GA']
enemy_list=[1,2,3,4,5,6,7,8]


for enemy in enemy_list:
    for method in method_list:
        
        env.update_parameter('enemies',[enemy])
        sol = find_best_solution(method, enemy)
        individual_gain_list = []
        for i in range(5):
            
            _, player_life, enemy_life, _=env.play(sol)
            indivdual_gain=player_life-enemy_life
            individual_gain_list.append(indivdual_gain)
        results[(enemy, method)] = individual_gain_list


fig, ax = plt.subplots(figsize=(24, 12))
data = [results[key] for key in sorted(results.keys())]
labels = [f"Method {m}, Enemy {e}" for m, e in sorted(results.keys())]

ax.boxplot(data, labels=labels)
ax.set_title('Individual Gains for Each Method and Enemy Combination', fontsize=20)
ax.set_xlabel('Method, Enemy Combination', fontsize=16)
ax.set_ylabel('Individual Gain', fontsize=16)
plt.xticks(rotation=45)
plt.savefig("box_plot.png", dpi=150)