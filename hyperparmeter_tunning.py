from skopt import Optimizer
from skopt.space import Real, Integer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re
from training_specialist_lingfeng import EvolutAlgorithmOptimizer  # 导入类 
from training_specialist_lingfeng import save_gain_results, collect_gain_results


def bayesian_optimization(experiment_name, enemies, mode="GA", n_calls=20):
  
    if mode == "GA":
        search_space = [
            Integer(5, 20, name='n_hidden_neurons'),   
            Integer(50, 300, name='n_population'),    
            Integer(20, 50, name='n_generations'),    
            Real(0.1, 0.3, name='mutation_rate')       
            #
        ]
    elif mode == "ES":
        search_space = [
            Integer(5, 20, name='n_hidden_neurons'),  
            Integer(50, 300, name='n_population'),    
            Integer(20, 50, name='n_generations'),    
            Real(0.1, 0.3, name='mutation_rate'),     
            Real(0.05, 0.2, name='sigma')            
        ]

    
    optimizer = Optimizer(search_space)  #initialize the optimizer  

    best_score = -float('inf')
    best_params = None
    all_results = []

    for i in range(n_calls):
        
        suggested_params = optimizer.ask()  #get the suggested hyperparameters

        # retrieve the hyperparameters
        if mode == "GA":
            n_hidden_neurons, n_population, n_generations, mutation_rate = suggested_params
            sigma = 0.1  # 
        elif mode == "ES":
            n_hidden_neurons, n_population, n_generations, mutation_rate, sigma = suggested_params

        print(f"Running trial {i + 1}/{n_calls} for mode {mode}")
        print(f"Hyperparameters: hidden_neurons={n_hidden_neurons}, population={n_population}, "
              f"generations={n_generations}, mutation_rate={mutation_rate}, sigma={sigma if mode == 'ES' else '0.1'}")

      
        optimizer_instance = EvolutAlgorithmOptimizer(
            experiment_name, enemies, n_hidden_neurons, n_population, n_generations,
            mutation_rate, sigma, mode=mode  # get the optimizer instance
        )
        optimizer_instance.execute()

        
        results = optimizer_instance.saveget_fitness()
        
        fitness = results['max_fitness'].item()   # get the max fitness value

        print(f"Fitness for this trial: {fitness}")

    
        all_results.append({
            'n_hidden_neurons': n_hidden_neurons,
            'n_population': n_population,
            'n_generations': n_generations,
            'mutation_rate': mutation_rate,
            'sigma': sigma if mode == "ES" else 0.1,  
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
                'sigma': sigma if mode == "ES" else 0.1  # GA模式下sigma为0.1
            }

    print(f"Best Hyperparameters for mode {mode}: {best_params}")
    print(f"Best Fitness: {best_score}")

    output_dir = f'{experiment_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(f'{output_dir}/bayes_search_results_{mode}.csv', index=False)
    print(f"Bayesian optimization results saved to {output_dir}/bayes_search_results_{mode}.csv")

    return best_params, best_score


if __name__ == "__main__":
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_parameter'

    enemies = [4]

    
    
    print("Running GA optimization:")
    best_params_ga, best_score_ga = bayesian_optimization(experiment_name, enemies, mode="GA", n_calls=20)
    print(f"Best Hyperparameters for GA: {best_params_ga}")
    

    print("\nRunning ES optimization:")
    best_params_es, best_score_es = bayesian_optimization(experiment_name, enemies, mode="ES", n_calls=20)
    print(f"Best Hyperparameters for ES: {best_params_es}")