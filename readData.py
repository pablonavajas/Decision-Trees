##############################################################################
# Dataset class to read in the attributes and labels from a file.            # 
##############################################################################

import numpy as np

class Dataset:

    def __init__(self, filename):
        self.attributes = self.import_data(filename)[0]
        self.labels = self.import_data(filename)[1]

    def import_data(self, filename):
        data = open(filename, 'r').read()

        L = data.split('\n')[:-1]

        J = [row[-1] for row in L]

        L = [row[:-2] for row in L]
        L = [i.split(",") for i in L]
        L = [[int(element) for element in row] for row in L]

        L = np.array(L)
        J = np.array(J)

        return L, J
