import random
import pandas as pd

# Define the neighbors for each node in a 3x3 grid
neighbors = {
    0: [0, 1, 3],
    1: [1, 0, 2, 4],
    2: [2, 1, 5],
    3: [3, 0, 4, 6],
    4: [4, 1, 3, 5, 7],
    5: [5, 2, 4, 8],
    6: [6, 3, 7],
    7: [7, 4, 6, 8],
    8: [8, 5, 7]
}

def generate_step1_step2(start, end):
    # Possible step1 nodes
    step1_options = neighbors[start]
    for step1 in step1_options:
        # Possible step2 nodes that are next to step1 and end
        # set(neighbors[step1]) & set(neighbors[end]) = same as in math. Finding common elements in two sets
        step2_options = list(set(neighbors[step1]) & set(neighbors[end]))
        if step2_options:
            step2 = random.choice(step2_options)
            return step1, step2
    
    # Case when there must be more than just 2 steps to reach the end node
    return -1, -1

def generate_dataset(n):
    dataset = []
    for _ in range(n):
        # Generating random start and end nodes
        start = random.randint(0, 8)
        end = random.randint(0, 8)
        while end == start:
            end = random.randint(0, 8)
        
        # Generating an input layer with the start and end nodes
        X = [[0], [0], [0],
            [0], [0], [0],
            [0], [0], [0]]
        X[start] = [1]
        X[end] = [1]
        

        step1, step2 = generate_step1_step2(start, end)
        dataset.append([X, step1, step2])
    return dataset

# Generate a dataset
dataset = generate_dataset(1200)
#print( dataset )

# Create a DataFrame
df = pd.DataFrame(dataset, columns=["X", "step1", "step2"])

# Write the DataFrame to an Excel file
df.to_excel("output.xlsx", index=False)