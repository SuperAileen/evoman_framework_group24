from skopt import Optimizer
from skopt.space import Real, Integer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re
from training_generalist import GeneralistOptimizer

def bayesian_optimization(experiment_name, enemy_set, mode="GA", n_calls=20):
  
    if mode == "GA1":
        search_space = [ 
            Integer(50, 300, name='n_population'),    
            Integer(30, 50, name='n_generations'),    
            Real(0.1, 0.3, name='mutation_rate'),
            Integer(2,5, name='tournament_size')       
            #
        ]

    elif mode == "GA2":
        search_space = [
            Integer(50, 300, name='n_population'),    
            Integer(30, 50, name='n_generations'),    
            Real(0.1, 0.3, name='mutation_rate'), 
            Integer(10, 30, name='k')      
        ]

    
    optimizer = Optimizer(search_space)  #initialize the optimizer  

    best_score = -float('inf')
    best_params = None
    all_results = []

    for i in range(n_calls):
        
        suggested_params = optimizer.ask()  #get the suggested hyperparameters

        # retrieve the hyperparameters
        if mode == "GA1":
            n_population, n_generations, mutation_rate = suggested_params
            k=10 
            sigma = 0.1  # 
            n_hidden_neurons=10
        elif mode == "GA2":
            n_population, n_generations, mutation_rate, k = suggested_params
            sigma = 0.1  #
            n_hidden_neurons=10

        print(f"Running trial {i + 1}/{n_calls} for mode {mode}")
        print(f"Hyperparameters:  population={n_population}, "
              f"generations={n_generations}, mutation_rate={mutation_rate}, k={k}")

      
        generalist_optimizer = GeneralistOptimizer(experiment_name, enemy_set, n_hidden_neurons, n_population, n_generations,
                                    mutation_rate, sigma, mode=mode, k=k, tournment_size=3)
        generalist_optimizer.execute()

        
        results = generalist_optimizer.saveget_fitness()
        
        fitness = results['max_fitness'].item()   # get the max fitness value

        print(f"Fitness for this trial: {fitness}")

    
        all_results.append({
            'n_population': n_population,
            'n_generations': n_generations,
            'mutation_rate': mutation_rate,
            'k': k if mode == "GA2" else 10,  
            'fitness': fitness
        })

     
        optimizer.tell(suggested_params, -fitness)  # use negative fitness for maximization

       
        if fitness > best_score:
            best_score = fitness
            best_params = {
                'n_hidden_neurons': n_hidden_neurons,
                'n_population': n_population,
                'n_generations': n_generations,
                'mutation_rate': mutation_rate,
                'k': k if mode == "GA2" else 10  # GA模式下sigma为0.1
            }

    #print(f"Best Hyperparameters for mode {mode}: {best_params}")
    #print(f"Best Fitness: {best_score}")
    

    output_dir = f'{experiment_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(f'{output_dir}/bayes_search_results_{mode}.csv', index=False)
    #print(f"Bayesian optimization results saved to {output_dir}/bayes_search_results_{mode}.csv")

    return best_params, best_score


if __name__ == "__main__":
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'generalist_parameter_optimization'

    enemy_set = [1,2,3,4,5,6,7,8]

    
    
    print("Running GA optimization:")
    best_params_ga1, best_score_ga1 = bayesian_optimization(experiment_name, enemy_set, mode="GA1", n_calls=2)
    print(f"Best Hyperparameters for GA: {best_params_ga1}")
    

    print("\nRunning ES optimization:")
    best_params_ga2, best_score_ga2 = bayesian_optimization(experiment_name, enemy_set, mode="GA2", n_calls=2)
    print(f"Best Hyperparameters for ES: {best_params_ga2}")

    