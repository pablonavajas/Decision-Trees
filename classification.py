##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np
import math


class DecisionTreeClassifier(object):
    """
    A decision tree classifier
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False
        self.node = None
    
    def train(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the 
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array
        
        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
      
        """ TEST
        node = self.find_best_node(x,y) 
        print(node.attribute)
        print(node.split_point)
        """
    
        #Trains the decision tree
        self.node = self.induce_decision_tree(x,y)

        print(self.node.child1.attribute)
        print(self.node.child1.split_point)
        print(self.node.child1.label)
        print(self.node.child2.label)

        #Saves the model to a file. """ CHECK """
        np.save('decision_tree.npy', self.node) 
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
        return self
    
    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        
        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """
        
        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        
        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
    
        # remember to change this if you rename the variable
        return predictions

        #######################################################################
        #                 ** HELPER FUNCTIONS **
        #######################################################################

    def induce_decision_tree(self, x, y):
        """
        Implementation of Algorithm 1 - Decision Tree Induction
        Returns the root node of the decision tree.
        """

        """ CHECK: "Dataset cannot be split further """
        #array of unique labels
        unique = np.unique(y)
        #find the best node (will test for info_gain next)
        node = self.find_best_node(x, y)

        #if all samples have the same label or there is no information to be 
        #gained by splitting
        if unique.size == 1 or node.info_gain == 0:
            return Node(label = unique[0])
        else:
            #node = self.find_best_node(x, y)
            children_datasets = self.split_dataset(x, y, node)
            
            """ DEBUG """
            #print("Attribute: ", node.attribute)
            #print("Split Point: ", node.split_point)
            #print("Children: ", children_datasets)

            for child_dataset in children_datasets:
                #print("Child")
                child_node = self.induce_decision_tree(child_dataset[0], child_dataset[1])
                node.add_child(child_node)
            
            return node
        

    def find_best_node(self, x, y):
        """
        Returns the best node based on x: an NxK NumPy array representing N training
        instances of K attributes and y: an N-dim NumPy array containing the class label
        for each N instance.
        """

        #initialise variables which store max info gain and
        #the attribute and split point
        max_information_gain = -1
        attribute = -1
        split_point = -1
        
        #get the size of the array x
        row_size, column_size = x.shape
        
        #loop through each column of X
        for col in range(column_size):

            #sort attributes and labels based on column
            x, y = self.sort(x, y, col)

            #Find possible splitting points and, for each, their information gain
            #(can start from 1 as we need to split)
            for row in range(1,row_size):
                #if x[row,col] is a splitting point
                """ CHECK """
                if x[row,col] != x[row-1,col]:
                    #slice the labels into two child sets
                    #child1 = [att_1, lab_1] = [x[:row], y[:row]]
                    #child2 = [att_2, lab_2] = [x[row:], y[row:]]
                    children = [child1, child2] = [y[:row], y[row:]]

                    #calculate information gain
                    info_gain = self.information_gain(y, children)

                    #test if this is max information gain so far
                    #if so, update holding values
                    if info_gain > max_information_gain:
                        max_information_gain = info_gain
                        attribute = col
                        split_point = x[row,col]

        #create node with attribute and value of max info_gain
        node = Node(attribute, split_point, max_information_gain)
        
        return node

    def sort(self, x, y, column):
        """
        Returns the array of attributes and the array of labels based on the column
        to be sorted.
        """
        indices = x[:,column].argsort()
        x = x[indices]
        y = y[indices]

        return x, y

    def information_gain(self, parent, children):
        """
        Calculates the information gain for a set of parent class labels, and a list
        of child sets (each set contains labels).
        """

        #total number of samples in the parent
        N = (parent.shape)[0] 
       
        #calculate the entropy of the parent
        h_parent = self.entropy(parent)

        #calculate the terms that when summed make up the second term in calculating IG
        h_children = [((child.shape)[0])*(1/N)*self.entropy(child) for child in children]

        h_bar_children = sum(h_children)

        return h_parent - h_bar_children

        
    def entropy(self, dataset):
        """
        Calculates the entropy for a given dataset.
        """

        #obtain the counts of each unique label
        unique, counts = np.unique(dataset, return_counts = True)
       
        #obtain an array of probabilities for each label
        total = sum(counts)
        probability = [i/total for i in counts]

        #get an array of terms for calculating entropy
        term = [i*math.log(i,2) for i in probability]

        #apply the entropy formula
        return -sum(term)


    def split_dataset(self, x, y, node):
        """
        Splits the dataset into child sets based on the node condition.
        """

        #Sort the attribute matrix and label matrix based on node.attribute
        x, y = self.sort(x, y, node.attribute)

        #print("X: ", x)
        #print("Y: ", y)

        #sp = node.split_point
        index = np.where(x[:,node.attribute] == node.split_point)
        sp = index[0][0]
        
        print("Split: ", sp)

        #Slice the datasets according to node.split_point
        child1 = [x[:sp], y[:sp]]
        child2 = [x[sp:], y[sp:]]
        children = [child1, child2]

        return children


class Node:
    """
    A Node object has the following data members:
        - the attribute and split point that results in the most informative split for
        some dataset (set to None if the Node is meant to contain only a label)
        - the label (set to None by default if the Node is meant to hold an attribute 
        and split_point)
        - the two child nodes that are pointed to by the parent (binary tree)

    It also has the add_child() method, which associates a child node with its parent
    """

    def __init__(self, attribute = None, split_point = None, info_gain = None, label = None):
        #column
        self.attribute = attribute
        #row
        self.split_point = split_point
        self.info_gain = info_gain
        self.label = label
        self.child1 = None
        self.child2 = None

    def add_child(self, child_node):
        if self.child1 == None:
            self.child1 = child_node
        elif self.child2 == None:
            self.child2 = child_node
        else:
            print("Error adding child_node")




