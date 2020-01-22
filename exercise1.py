##############################################################################
# CO395: Introduction to Machine Learning                                    #
# Coursework 1 data loading code                                             #
# Prepared by: Pablo Navajas Helguero                                        #
##############################################################################

import numpy as np

#ie: filename = 'data/simple1.txt'

def load_data(filename):

    x = np.array([])

    a = np.array([])

    data = open(filename,'r+');

    for line in data:

        line = line.rstrip()
    
        arr = [int(x) for x in line.split(',')[:-1]]

        y = np.array(arr)

        if (x.size > 0):
            x = np.vstack((x,y))
        else:
            x = y

        b = np.array(line.split(',')[-1])

        if (a.size > 0):
            a = np.append(a,b)
        else:
            a = b
            
    data.close()

    print(x)
    print(a)

    return 0



if __name__ == "__main__":

    load_data('data/simple1.txt')
