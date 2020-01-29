##############################################################################
# CO395: Introduction to Machine Learning                                    #
# Coursework 1 data loading code                                             #
# Prepared by: EPC                                                           #
# Tests: the class written in test_class_exercise1.py                        #
##############################################################################

import test_class_exercise1 as DS
import numpy as np
import math
import classification as cl

#data = DS.Dataset('data/simple1.txt')
data = DS.Dataset('data/toy.txt')

print(data.attributes)
print(data.labels)

sortedArr = data.attributes[data.attributes[:,1].argsort()]
sortedLabels = data.labels[data.attributes[:,1].argsort()]

row, col = sortedArr.shape

print(sortedArr[:,1])
a = sortedArr[:,1]
a = np.where(a == 6)
index = a[0][0]
print(index)
print(sortedArr)
print(sortedLabels)
"""
for column in range(1, col):
    print(column)


print((sortedArr.shape)[0])

unique, counts = np.unique(sortedLabels, return_counts = True)
print(np.unique(sortedLabels, return_counts = True))
print(unique.size)
print(unique[0])
print(counts)
total = sum(counts)
#probability = [i/total for i in counts]
#term = [i*math.log(i,2) for i in probability]
#print(probability)
#print(term)
#print(-sum(term))
#print(np.asarray((unique, counts)).T)

node = cl.Node(label = 'C')
print(node.label)
print(node.attribute)

"""
"""
point = 3

#a_child1 = sortedArr[:point]
l_child1 = sortedLabels[:point]

#child1 = [a_1, l_1] = [sortedArr[:point], sortedLabels[:point]]

#a_child2 = sortedArr[point:]
l_child2 = sortedLabels[point:]

#print(sortedArr)
#print(sortedLabels)

print(l_child1)
print(l_child2)
"""
"""
print(a_child2)
print(l_child2)

print(child1)
print(child1[0])
print(child1[1])
print(a_1)
print(l_1)

print([data.attributes,data.labels])
"""
