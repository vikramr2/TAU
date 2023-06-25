import numpy as np
import tau2

g = tau2.load_graph('example.graph')

# data = np.load('TAU_partition_example.graph.npy')
# Main parameter values: pop_size=60, workers=60, max_generations=500
# Main parameter values: pop_size=60, workers=60, max_generations=500
data = tau2.tau(g)

print(data)
print(len(data))
print(list(zip(list(range(1,len(data)+1)), data)))