import numpy as np
import src.tau as tau

g = tau.load_graph('example.graph')

# data = np.load('TAU_partition_example.graph.npy')
# Main parameter values: pop_size=60, workers=60, max_generations=500
# Main parameter values: pop_size=60, workers=60, max_generations=500
data = tau.tau(g)

print(data)
print(len(data))
print(list(zip(list(range(1,len(data)+1)), data)))