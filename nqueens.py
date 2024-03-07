import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):

   # Initialize counter
    fitness_cnt = 0

        # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt

fitness = mlrose.CustomFitness(queens_max)
fitness2 = mlrose.CustomFitness(queens_max)
fitness3 = mlrose.CustomFitness(queens_max)

# problem = mlrose.DiscreteOpt(length = 4, fitness_fn = fitness, maximize = False, max_val = 8)
problem = mlrose.QueensOpt(length = 8, fitness_fn = fitness, maximize = True)
problem2 = mlrose.QueensOpt(length = 16,fitness_fn = fitness2, maximize = True)
problem3 = mlrose.QueensOpt(length = 32, fitness_fn = fitness3, maximize = True)


def performNQueensGA(problem, pop_size, mutation_prob):
    best_ga_state = mlrose.genetic_alg(problem=problem, max_iters = 5000, max_attempts = 100, pop_size = pop_size, mutation_prob=mutation_prob, random_state=60, curve=True)

    print("Max Knapsack fitness found using the Genetic Algorithm was: " + str(best_ga_state[1]))
    print(best_ga_state[0])
    # print(best_ga_state[2])

    print()

def performNQueensSA(problem, T):

    best_ga_state = mlrose.simulated_annealing(problem=problem, max_iters = 5000, max_attempts = 50, schedule=mlrose.GeomDecay(init_temp=T), random_state=60, curve=True)

    print("Max Knapsack fitness found using the Simulated Annealing Algorithm was: " + str(best_ga_state[1]))
    print(best_ga_state[0])
    # print(best_ga_state[2][:,0])

    print()

def performNQueensRHC(problem, restarts):

    best_rhc_state = mlrose.random_hill_climb(problem=problem, max_iters = 5000, max_attempts = 500, restarts=restarts, random_state=60, curve=True)

    print("Max Knapsack fitness found using the Random Hill Climbing Algorithm was: " + str(best_rhc_state[1]))
    print(best_rhc_state[0])
    # print(best_rhc_state[2])

    print()

def performNQueensMIMIC(problem, pop_size, keep_pct):

    best_ga_state = mlrose.mimic(problem=problem, max_iters = 5000, max_attempts = 100, pop_size = pop_size, keep_pct=keep_pct, random_state=60, curve=True)

    print("Max Knapsack fitness found using the MIMIC Algorithm was: " + str(best_ga_state[1]))
    print(best_ga_state[0])
    # print(best_ga_state[2][:,0])

    print()

def testPopGeneticValidation(problem, problem2, problem3, pop_sizes, random_seeds):
    problems = [
        problem,
        problem2, 
        problem3
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

def testMutationGeneticValidation(problem, problem2, problem3, mutation_probs, random_seeds):
    problems = [
        problem,
        problem2, 
        problem3
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
                    problem=problem, max_iters=5000, max_attempts=100, 
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

def testTemperatureSAValidation(problem, problem2, problem3, temperatures, random_seeds):
    problems = [
        problem,
        problem2, 
        problem3
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
                    problem=problem, max_iters=5000, max_attempts=100, 
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

def testPopMIMICValidation(problem, problem2, problem3, pop_sizes, random_seeds):
    problems = [
        problem,
        problem2, 
        problem3
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

def testKeepMIMICValidation(problem, problem2, problem3, keep_pcts, random_seeds):
    problems = [
        problem,
        problem2, 
        problem3
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

def testRestartsRHCValidation(problem, problem2, problem3, restarts_list):
    problems = [
        problem,
        problem2, 
        problem3
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
                    problem=problem, max_iters=5000, max_attempts=100, 
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

# testPopGeneticValidation(problem, problem2, problem3, pop_sizes, random_seeds)
# testMutationGeneticValidation(problem, problem2, problem3, mutation_probs, random_seeds)
start_time = time.time()
performNQueensGA(problem, 30, 0.1)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensGA(problem2, 30, 0.1)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensGA(problem3, 80, 0.2)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")






# Simulated Annealing
schedules = [mlrose.GeomDecay, mlrose.ExpDecay, mlrose.ArithDecay]
temperatures = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

start_time = time.time()
performNQueensSA(problem, 1)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensSA(problem2, 5)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensSA(problem3, 20)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")






# testTemperatureSAValidation(problem, problem2, problem3, temperatures, random_seeds)

# MIMIC
pop_sizes = [100, 125, 150, 175, 200, 250, 300]
keep_pcts = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

start_time = time.time()
performNQueensMIMIC(problem, 250, 0.2)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensMIMIC(problem2, 300, 0.3)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensMIMIC(problem3, 500, 0.4)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")






# testPopMIMICValidation(problem, problem2, problem3, pop_sizes, random_seeds)
# testKeepMIMICValidation(problem, problem2, problem3, keep_pcts, random_seeds)

# Random Hill Climb
restarts_list = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]

start_time = time.time()
performNQueensRHC(problem, 5)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensRHC(problem2, 70)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

start_time = time.time()
performNQueensRHC(problem3, 80)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")






# testRestartsRHCValidation(problem, problem2, problem3, restarts_list)
















