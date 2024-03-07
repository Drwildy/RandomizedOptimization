import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
from nqueens import queens_max
import time


def knapsack():
    fitness = mlrose.Knapsack(weights=[7, 2, 1, 9], values=[5, 4, 5, 15], max_weight_pct=.5)
    fitness2 = mlrose.Knapsack(weights=[1, 10, 6, 9, 4, 6, 10, 3], values=[7, 8, 9, 9, 9, 5, 3, 4], max_weight_pct=.35)
    fitness3 = mlrose.Knapsack(weights=[6, 2, 1, 3, 3, 6, 6, 1, 6, 6, 2, 9, 4, 8, 4, 9], values=[10, 2, 4, 3, 9, 2, 4, 3, 2, 3, 4, 1, 1, 3, 4, 5], max_weight_pct=.35)

    problem = mlrose.KnapsackOpt(fitness_fn = fitness, maximize = True)
    problem2 = mlrose.KnapsackOpt(fitness_fn = fitness2, maximize = True)
    problem3 = mlrose.KnapsackOpt(fitness_fn = fitness3, maximize = True)

    # Initialize lists to store best curves for each problem
    best_curves = []

    # List of problems to iterate over
    problems = [problem, problem2, problem3]

    # Corresponding number of restarts for each problem
    restarts_list = [5, 10, 300]

    for prob, restarts in zip(problems, restarts_list):
        rhc = mlrose.RHCRunner(problem=prob,
                            experiment_name='runner practice',
                            output_directory=None,
                            seed=60,
                            iteration_list=[10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 2000, 3000, 4000, 5000],
                            max_attempts=100,
                            restart_list=[restarts])
        df_run_stats, df_run_curves = rhc.run()
        best_fitness = df_run_curves['Fitness'].max()
        best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
        best_restart = best_runs[best_runs['Iteration'] == best_runs['Iteration'].min()].iloc[0]
        best_curve = df_run_curves[df_run_curves['current_restart'] == best_restart['current_restart']]
        best_curves.append(best_curve)

    # Now plot all curves on the same graph
    plt.figure(figsize=(10, 6))

    # Colors or markers for each problem's curve
    colors = ['r', 'g', 'b']
    labels = ['Problem 1', 'Problem 2', 'Problem 3']

    for i, curve in enumerate(best_curves):
        plt.plot(curve["Iteration"], curve["Fitness"], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Fitness Score vs. Iteration For Each Knapsack Problem')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))

    # Colors or markers for each problem's curve
    colors = ['r', 'g', 'b']
    labels = ['Problem 1', 'Problem 2', 'Problem 3']

    for i, curve in enumerate(best_curves):
        plt.plot(curve["Iteration"], curve["FEvals"], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('FEvals vs. Iteration For Each Problem')
    plt.xlabel('Iteration')
    plt.ylabel('FEvals')
    plt.legend()
    plt.grid(True)

    plt.show()

def nQueens():
    fitness = mlrose.CustomFitness(queens_max)
    fitness2 = mlrose.CustomFitness(queens_max)
    fitness3 = mlrose.CustomFitness(queens_max)

    # problem = mlrose.DiscreteOpt(length = 4, fitness_fn = fitness, maximize = False, max_val = 8)
    problem = mlrose.QueensOpt(length = 8, fitness_fn = fitness, maximize = True)
    problem2 = mlrose.QueensOpt(length = 16,fitness_fn = fitness2, maximize = True)
    problem3 = mlrose.QueensOpt(length = 32, fitness_fn = fitness3, maximize = True)

    # Initialize lists to store best curves for each problem
    best_curves = []

    # List of problems to iterate over
    problems = [problem, problem2, problem3]

    # Corresponding number of restarts for each problem
    restarts_list = [5, 70, 80]

    for prob, restarts in zip(problems, restarts_list):
        rhc = mlrose.RHCRunner(problem=prob,
                            experiment_name='runner practice',
                            output_directory=None,
                            seed=60,
                            iteration_list=[10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 2000, 3000, 4000, 5000],
                            max_attempts=100,
                            restart_list=[restarts])
        df_run_stats, df_run_curves = rhc.run()
        best_fitness = df_run_curves['Fitness'].max()
        best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
        best_restart = best_runs[best_runs['Iteration'] == best_runs['Iteration'].min()].iloc[0]
        best_curve = df_run_curves[df_run_curves['current_restart'] == best_restart['current_restart']]
        best_curves.append(best_curve)

    # Now plot all curves on the same graph
    plt.figure(figsize=(10, 6))

    # Colors or markers for each problem's curve
    colors = ['r', 'g', 'b']
    labels = ['Problem 1', 'Problem 2', 'Problem 3']

    for i, curve in enumerate(best_curves):
        plt.plot(curve["Iteration"], curve["Fitness"], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Fitness Score vs. Iteration For Each NQueens Problem')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))

    # Colors or markers for each problem's curve
    colors = ['r', 'g', 'b']
    labels = ['Problem 1', 'Problem 2', 'Problem 3']

    for i, curve in enumerate(best_curves):
        plt.plot(curve["Iteration"], curve["FEvals"], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('FEvals vs. Iteration For Each Problem')
    plt.xlabel('Iteration')
    plt.ylabel('FEvals')
    plt.legend()
    plt.grid(True)

    plt.show()

def fourPeaks():
    fitness = mlrose.FourPeaks(t_pct=0.3)
    fitness2 = mlrose.FourPeaks(t_pct=0.3)
    fitness3 = mlrose.FourPeaks(t_pct=0.3)

    # problem = mlrose.DiscreteOpt(length = 4, fitness_fn = fitness, maximize = False, max_val = 8)
    problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = True)
    problem2 = mlrose.DiscreteOpt(length = 16,fitness_fn = fitness2, maximize = True)
    problem3 = mlrose.DiscreteOpt(length = 32, fitness_fn = fitness3, maximize = True)

    # Initialize lists to store best curves for each problem
    best_curves = []

    # List of problems to iterate over
    problems = [problem, problem2, problem3]

    # Corresponding number of restarts for each problem
    restarts_list = [20, 70, 100]

    for prob, restarts in zip(problems, restarts_list):
        rhc = mlrose.RHCRunner(problem=prob,
                            experiment_name='runner practice',
                            output_directory=None,
                            seed=60,
                            iteration_list=[10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 2000, 3000, 4000, 5000],
                            max_attempts=100,
                            restart_list=[restarts])
        df_run_stats, df_run_curves = rhc.run()
        best_fitness = df_run_curves['Fitness'].max()
        best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
        best_restart = best_runs[best_runs['Iteration'] == best_runs['Iteration'].min()].iloc[0]
        best_curve = df_run_curves[df_run_curves['current_restart'] == best_restart['current_restart']]
        best_curves.append(best_curve)

    # Now plot all curves on the same graph
    plt.figure(figsize=(10, 6))

    # Colors or markers for each problem's curve
    colors = ['r', 'g', 'b']
    labels = ['Problem 1', 'Problem 2', 'Problem 3']

    for i, curve in enumerate(best_curves):
        plt.plot(curve["Iteration"], curve["Fitness"], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('Fitness Score vs. Iteration For Each Four Peaks Problem')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))

    # Colors or markers for each problem's curve
    colors = ['r', 'g', 'b']
    labels = ['Problem 1', 'Problem 2', 'Problem 3']

    for i, curve in enumerate(best_curves):
        plt.plot(curve["Iteration"], curve["FEvals"], marker='o', linestyle='-', color=colors[i], label=labels[i])

    plt.title('FEvals vs. Iteration For Each Problem')
    plt.xlabel('Iteration')
    plt.ylabel('FEvals')
    plt.legend()
    plt.grid(True)

    plt.show()


start_time = time.time()
knapsack()
# nQueens()
# fourPeaks()
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")