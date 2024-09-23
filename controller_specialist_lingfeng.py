#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'optimization_train_lingfeng_m1_20240923-184821'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)

enemy_list=[1,2,3]


# tests saved demo solutions for each enemy
for en in enemy_list:
      

	#Update the enemy
	env.update_parameter('enemies',[en])

	# Load specialist controller
	sol = np.loadtxt(experiment_name+'/best.txt')
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
	fitness, player_life, enemy_life, time_taken=env.play(sol)
	print(f"Fitness: {fitness}, Player Life: {player_life}, Enemy Life: {enemy_life}, Time Taken: {time_taken}")

print("Done testing all enemies.")
# print(fitness, player_life, enemy_life, time_taken)


'''
# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)
'''
