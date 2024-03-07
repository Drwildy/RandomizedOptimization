import random

def generate_knapsack_problem(size):
    """
    Generates random weights and values for a knapsack problem of a given size.
    
    Parameters:
    - size (int): The number of items to generate.
    
    Returns:
    - tuple of two lists: (weights, values)
    """
    weights = [random.randint(1, 10) for _ in range(size)]
    values = [random.randint(1, 10) for _ in range(size)]
    return weights, values

# Example usage
size = 16
weights, values = generate_knapsack_problem(size)
print(weights, values)
