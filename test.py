import numpy as np

a = np.array([[1,2,3],
              [2,3,4],
              [1,6,5], 
              [9,3,4]])
a = a[:,[1,0,2]]
print(a)