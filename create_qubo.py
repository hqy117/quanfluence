import dimod
import numpy as np

# Create a simple QUBO problem

# Create a simple graph (a triangle with weights)
edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]  # (node1, node2, weight)

# Initialize the QUBO dictionary
Q = {}

# Add terms to the QUBO
for i, j, weight in edges:
    # Add the quadratic terms for each edge
    if (i, j) in Q:
        Q[(i, j)] += 2 * weight
    else:
        Q[(i, j)] = 2 * weight
    
    # Add the corresponding linear terms
    if (i, i) in Q:
        Q[(i, i)] -= weight
    else:
        Q[(i, i)] = -weight
        
    if (j, j) in Q:
        Q[(j, j)] -= weight
    else:
        Q[(j, j)] = -weight

# Create a BQM (Binary Quadratic Model) from the QUBO dictionary
bqm = dimod.BQM.from_qubo(Q)

# Save the QUBO to a file
output_file = "max_cut_triangle.qubo"
with open(output_file, 'w') as f:
    # Use a different approach to save the QUBO file
    f.write("p qubo 0 {} {}\n".format(bqm.num_variables, len(Q)))
    for (i, j), value in Q.items():
        f.write("{} {} {}\n".format(i, j, value))

print(f"QUBO problem saved to {output_file}")
print("Q matrix:")
for (i, j), value in Q.items():
    print(f"Q[{i},{j}] = {value}")