import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# --- 1. Graph & Path Setup ---
edges = {
    'A-B': 1.0, 'A-C': 1.3, 'B-D': 1.1, 'C-D': 0.9, 'D-E': 1.2,
    'E-F': 1.0, 'E-G': 1.5, 'F-H': 1.2, 'G-H': 1.1, 'C-F': 1.4, 'B-G': 1.6
}

paths = [
    'A-B-D-E-F-H',  # path0
    'A-B-D-E-G-H',  # path1
    'A-C-D-E-F-H',  # path2
    'A-C-D-E-G-H',  # path3
    'A-C-F-H',      # path4
    'A-B-G-H'       # path5
]

def path_cost(path_str: str) -> float:
    nodes = path_str.split('-')
    return sum(edges[f"{nodes[k]}-{nodes[k+1]}"] for k in range(len(nodes) - 1))

path_costs = [path_cost(p) for p in paths]
num_paths = len(paths)

# --- 2. Build QUBO Matrix (Single-Path Selection Penalty) ---
P = 10.0  # Penalty weight for selecting anything other than exactly 1 path
qubo_matrix = np.zeros((num_paths, num_paths))

for i in range(num_paths):
    qubo_matrix[i, i] = path_costs[i] - P

for i in range(num_paths):
    for j in range(i + 1, num_paths):
        qubo_matrix[i, j] = 2 * P
        qubo_matrix[j, i] = 2 * P

# --- 3. Create QuadraticProgram ---
qp = QuadraticProgram(name="Shortest Path Selection")
for i in range(num_paths):
    qp.binary_var(name=f"x{i}")

linear = {f"x{i}": qubo_matrix[i, i] for i in range(num_paths)}
quadratic = {
    (f"x{i}", f"x{j}"): qubo_matrix[i, j]
    for i in range(num_paths)
    for j in range(i + 1, num_paths)
    if qubo_matrix[i, j] != 0
}

qp.minimize(linear=linear, quadratic=quadratic)

# --- 4. QAOA Solver Setup (Qiskit 1.0+ Compatible) ---
sampler = StatevectorSampler()
cobyla = COBYLA(maxiter=100)
qaoa = QAOA(sampler=sampler, optimizer=cobyla, reps=2)
optimizer = MinimumEigenOptimizer(qaoa)

# --- 5. Solve & Display Results ---
result = optimizer.solve(qp)

print("Optimal path selection (binary vector):", result.x)
print("Objective value:", result.fval)

chosen_paths = [paths[i] for i, b in enumerate(result.x) if b == 1]
print("Chosen path(s):", chosen_paths)
