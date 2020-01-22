#EPC test for reading data

import numpy as np

data = open('data/toy.txt', 'r').read()

L = data.split('\n')[:-1]

J = [row[-1] for row in L]

L = [row[:-2] for row in L]
L = [i.split(",") for i in L]
L = [[int(element) for element in row] for row in L]

L = np.array(L)
J = np.array(J)

print(L)
print(J)

