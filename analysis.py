import numpy as np
from tau import TAU

data = np.load('TAU_partition_example.graph.npy')
print(data)
print(len(data))
print(list(zip(list(range(1,len(data)+1)), data)))