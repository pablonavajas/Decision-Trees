##############################################################################
# CO395: Introduction to Machine Learning                                    #
# Coursework 1 data loading code                                             #
# Prepared by: EPC                                                           #
# Tests: the class written in test_class_exercise1.py                        #
##############################################################################

import test_class_exercise1 as DS

data = DS.Dataset('data/simple1.txt')

print(data.attributes)
print(data.labels)
