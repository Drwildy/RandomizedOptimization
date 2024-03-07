import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score


def performKnapsackGA(fitness, pop_size, mutation_prob):

    problem = mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True)
    best_ga_state = mlrose.genetic_alg(problem=problem, max_iters = 5000, max_attempts = 50, pop_size = pop_size, mutation_prob=mutation_prob, random_state=60, curve=True)

    print("Max Knapsack fitness found using the Genetic Algorithm was: " + str(best_ga_state[1]))
    print(best_ga_state[0])
    print(best_ga_state[2])

    print()

def performKnapsackSA(fitness, T):

    problem = mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True)
    best_ga_state = mlrose.simulated_annealing(problem=problem, max_iters = 5000, max_attempts = 50, schedule=mlrose.GeomDecay(init_temp=T), random_state=60, curve=True)

    print("Max Knapsack fitness found using the Simulated Annealing Algorithm was: " + str(best_ga_state[1]))
    print(best_ga_state[0])
    # print(best_ga_state[2][:,0])

    print()

def performKnapsackRHC(fitness, restarts):

    problem = mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True)
    best_rhc_state = mlrose.random_hill_climb(problem=problem, max_iters = 5000, max_attempts = 500, restarts=restarts, random_state=60, curve=True)

    print("Max Knapsack fitness found using the Random Hill Climbing Algorithm was: " + str(best_rhc_state[1]))
    print(best_rhc_state[0])
    print(best_rhc_state[2])

    print()


def performKnapsackMIMIC(fitness, pop_size, keep_pct):

    problem = mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True)
    best_ga_state = mlrose.mimic(problem=problem, max_iters = 5000, max_attempts = 500, pop_size = pop_size, keep_pct=keep_pct, random_state=60, curve=True)

    print("Max Knapsack fitness found using the MIMIC Algorithm was: " + str(best_ga_state[1]))
    print(best_ga_state[0])
    # print(best_ga_state[2][:,0])

    print()

def testPopGeneticValidation(fitness, fitness2, fitness3, pop_sizes, random_seeds):
    problems = [
        mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness2, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness3, maximize = True),
    ]

    # List to store mean fitness scores for each problem
    mean_fitness_scores = [[] for _ in problems]

    for index, problem in enumerate(problems):
        # Loop over each pop_size value
        for pop_size in pop_sizes:
            # Temporary list to store fitness scores for the current pop_size across all random seeds
            temp_fitness_scores = []

            # Loop over each random seed
            for seed in random_seeds:
                # Run genetic algorithm with the current pop_size and random seed
                best_state, best_fitness, bestcurve = mlrose.genetic_alg(
                    problem=problem, max_iters=5000, max_attempts=100, 
                    pop_size=pop_size, random_state=seed
                )
                # Append the fitness score to the temporary list
                temp_fitness_scores.append(best_fitness)

            # Calculate the mean fitness score for the current pop_size and append it to the list for the current problem
            mean_fitness_scores[index].append(np.mean(temp_fitness_scores))

    # Plotting the validation curves
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Different colors for each problem
    labels = ['Problem 1', 'Problem 2', 'Problem 3']
    
    for i in range(len(problems)):
        plt.plot(pop_sizes, mean_fitness_scores[i], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Mean Validation Curve for pop_size Across Problems')
    plt.xlabel('Population Size (pop_size)')
    plt.ylabel('Mean Best Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def testMutationGeneticValidation(fitness, fitness2, fitness3, mutation_probs, random_seeds):
    problems = [
        mlrose.KnapsackOpt(fitness_fn=fitness, maximize=True),
        mlrose.KnapsackOpt(fitness_fn=fitness2, maximize=True),
        mlrose.KnapsackOpt(fitness_fn=fitness3, maximize=True),
    ]

    # List to store mean fitness scores for each problem
    mean_fitness_scores = [[] for _ in problems]

    # Loop over each problem
    for index, problem in enumerate(problems):
        # Loop over each mutation probability
        for mutation_prob in mutation_probs:
            # Temporary list to store fitness scores for the current mutation probability
            temp_fitness_scores = []
            
            # Loop over each random seed
            for seed in random_seeds:
                # Run genetic algorithm with the current mutation probability and random seed
                _, best_fitness, _ = mlrose.genetic_alg(
                    problem=problem, max_iters=5000, max_attempts=50, 
                    pop_size=20, random_state=seed, mutation_prob=mutation_prob
                )
                # Append the fitness score to the temporary list
                temp_fitness_scores.append(best_fitness)
            
            # Calculate the mean fitness score for the current mutation probability
            mean_fitness = np.mean(temp_fitness_scores)
            # Append the mean fitness score to the list for the current problem
            mean_fitness_scores[index].append(mean_fitness)

    # Plotting the validation curves
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Different colors for each problem
    labels = ['Problem 1', 'Problem 2', 'Problem 3']
    
    for i in range(len(problems)):
        plt.plot(mutation_probs, mean_fitness_scores[i], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Mean Validation Curve for mutation_prob Across Problems')
    plt.xlabel('Mutation Probability')
    plt.ylabel('Mean Best Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def testTemperatureSAValidation(fitness, fitness2, fitness3, temperatures, random_seeds):
    problems = [
        mlrose.KnapsackOpt(fitness_fn=fitness, maximize=True),
        mlrose.KnapsackOpt(fitness_fn=fitness2, maximize=True),
        mlrose.KnapsackOpt(fitness_fn=fitness3, maximize=True),
    ]

    # List to store mean fitness scores for each problem
    mean_fitness_scores = [[] for _ in problems]

    # Loop over each problem
    for index, problem in enumerate(problems):
        # Loop over each mutation probability
        for t in temperatures:
            # Temporary list to store fitness scores for the current mutation probability
            temp_fitness_scores = []
            
            # Loop over each random seed
            for seed in random_seeds:

                _, best_fitness, _ = mlrose.simulated_annealing(
                    problem=problem, max_iters=5000, max_attempts=50, 
                    random_state=seed, schedule=mlrose.GeomDecay(init_temp=t)
                )
                # Append the fitness score to the temporary list
                temp_fitness_scores.append(best_fitness)
            
            # Calculate the mean fitness score for the current mutation probability
            mean_fitness = np.mean(temp_fitness_scores)
            # Append the mean fitness score to the list for the current problem
            mean_fitness_scores[index].append(mean_fitness)

    # Plotting the validation curves
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Different colors for each problem
    labels = ['Problem 1', 'Problem 2', 'Problem 3']
    
    for i in range(len(problems)):
        plt.plot(temperatures, mean_fitness_scores[i], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Mean Validation Curve for Temperature Across Problems')
    plt.xlabel('Temperature')
    plt.ylabel('Mean Best Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def testPopMIMICValidation(fitness, fitness2, fitness3, pop_sizes, random_seeds):
    problems = [
        mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness2, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness3, maximize = True),
    ]

    # List to store mean fitness scores for each problem
    mean_fitness_scores = [[] for _ in problems]

    for index, problem in enumerate(problems):
        # Loop over each pop_size value
        for pop_size in pop_sizes:
            # Temporary list to store fitness scores for the current pop_size across all random seeds
            temp_fitness_scores = []

            # Loop over each random seed
            for seed in random_seeds:
                # Run genetic algorithm with the current pop_size and random seed
                best_state, best_fitness, bestcurve = mlrose.mimic(
                    problem=problem, max_iters=5000, max_attempts=50, 
                    pop_size=pop_size, random_state=seed
                )
                # Append the fitness score to the temporary list
                temp_fitness_scores.append(best_fitness)

            # Calculate the mean fitness score for the current pop_size and append it to the list for the current problem
            mean_fitness_scores[index].append(np.mean(temp_fitness_scores))

    # Plotting the validation curves
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Different colors for each problem
    labels = ['Problem 1', 'Problem 2', 'Problem 3']
    
    for i in range(len(problems)):
        plt.plot(pop_sizes, mean_fitness_scores[i], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Mean Validation Curve for pop_size Across Problems')
    plt.xlabel('Population Size (pop_size)')
    plt.ylabel('Mean Best Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def testKeepMIMICValidation(fitness, fitness2, fitness3, keep_pcts, random_seeds):
    problems = [
        mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness2, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness3, maximize = True),
    ]

    # List to store mean fitness scores for each problem
    mean_fitness_scores = [[] for _ in problems]

    for index, problem in enumerate(problems):
        # Loop over each pop_size value
        for keep_pct in keep_pcts:
            # Temporary list to store fitness scores for the current pop_size across all random seeds
            temp_fitness_scores = []

            # Loop over each random seed
            for seed in random_seeds:
                # Run genetic algorithm with the current pop_size and random seed
                best_state, best_fitness, bestcurve = mlrose.mimic(
                    problem=problem, max_iters=5000, max_attempts=100, 
                    pop_size=50, keep_pct=keep_pct, random_state=seed
                )
                # Append the fitness score to the temporary list
                temp_fitness_scores.append(best_fitness)

            # Calculate the mean fitness score for the current pop_size and append it to the list for the current problem
            mean_fitness_scores[index].append(np.mean(temp_fitness_scores))

    # Plotting the validation curves
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Different colors for each problem
    labels = ['Problem 1', 'Problem 2', 'Problem 3']
    
    for i in range(len(problems)):
        plt.plot(keep_pcts, mean_fitness_scores[i], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Mean Validation Curve for keep_pct Across Problems')
    plt.xlabel('Keep Percentage')
    plt.ylabel('Mean Best Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def testRestartsRHCValidation(fitness, fitness2, fitness3, restarts_list):
    problems = [
        mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness2, maximize = True),
        mlrose.KnapsackOpt(fitness_fn = fitness3, maximize = True),
    ]

    # List to store mean fitness scores for each problem
    mean_fitness_scores = [[] for _ in problems]

    for index, problem in enumerate(problems):
        # Loop over each pop_size value
        for restarts in restarts_list:
            # Temporary list to store fitness scores for the current pop_size across all random seeds
            temp_fitness_scores = []

            # Loop over each random seed
            for seed in random_seeds:
                # Run genetic algorithm with the current pop_size and random seed
                best_state, best_fitness, bestcurve = mlrose.random_hill_climb(
                    problem=problem, max_iters=5000, max_attempts=50, 
                    restarts=restarts, random_state=seed
                )
                # Append the fitness score to the temporary list
                temp_fitness_scores.append(best_fitness)

            # Calculate the mean fitness score for the current pop_size and append it to the list for the current problem
            mean_fitness_scores[index].append(np.mean(temp_fitness_scores))

    # Plotting the validation curves
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Different colors for each problem
    labels = ['Problem 1', 'Problem 2', 'Problem 3']
    
    for i in range(len(problems)):
        plt.plot(restarts_list, mean_fitness_scores[i], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Mean Validation Curve for restarts Across Problems')
    plt.xlabel('Restarts')
    plt.ylabel('Mean Best Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.show()


fitness = mlrose.Knapsack(weights=[7, 2, 1, 9], values=[5, 4, 5, 15], max_weight_pct=.5)
fitness2 = mlrose.Knapsack(weights=[1, 10, 6, 9, 4, 6, 10, 3], values=[7, 8, 9, 9, 9, 5, 3, 4], max_weight_pct=.35)
fitness3 = mlrose.Knapsack(weights=[6, 2, 1, 3, 3, 6, 6, 1, 6, 6, 2, 9, 4, 8, 4, 9], values=[10, 2, 4, 3, 9, 2, 4, 3, 2, 3, 4, 1, 1, 3, 4, 5], max_weight_pct=.35)

# Define a range of variables to use
random_seeds= [20,40,60,80,100]

# Genetic Algorithm
pop_sizes = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mutation_probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# testPopGeneticValidation(fitness, fitness2, fitness3, pop_sizes, random_seeds)

# testMutationGeneticValidation(fitness, fitness2, fitness3, mutation_probs, random_seeds)
start_time = time.time()
performKnapsackGA(fitness, 10, 0.1)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackGA(fitness2, 70, 0.1)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackGA(fitness3, 50, 0.2)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")





# Simulated Annealing
schedules = [mlrose.GeomDecay, mlrose.ExpDecay, mlrose.ArithDecay]
temperatures = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]

start_time = time.time()
performKnapsackSA(fitness, 15)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackSA(fitness2, 5)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackSA(fitness3, 30)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# testTemperatureSAValidation(fitness, fitness2, fitness3, temperatures, random_seeds)

# MIMIC
keep_pcts = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

start_time = time.time()
performKnapsackMIMIC(fitness, 10, 0.1)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackMIMIC(fitness2, 70, 0.3)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackMIMIC(fitness3, 100, 0.2)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")


# testPopMIMICValidation(fitness, fitness2, fitness3, pop_sizes, random_seeds)
# testKeepMIMICValidation(fitness, fitness2, fitness3, keep_pcts, random_seeds)

# Random Hill Climb
restarts_list = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]

start_time = time.time()
performKnapsackRHC(fitness, 5)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackRHC(fitness2, 10)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performKnapsackRHC(fitness3, 300)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")





# testRestartsRHCValidation(fitness, fitness2, fitness3, restarts_list)
















