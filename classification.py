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
import eval

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

        #Trains the decision tree
        self.node = self.induce_decision_tree(x,y)

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

        for row in range(x.shape[0]):
            predictions[row] = self.check_decision_tree(x[row], self.node)

        return predictions

        #######################################################################
        #                 ** HELPER FUNCTIONS Q2 **
        #######################################################################

    #def induce_decision_tree(self, x, y):
    def induce_decision_tree(self, x, y, dep = 0):
        """
        Implementation of Algorithm 1 - Decision Tree Induction
        Returns the root node of the decision tree.
        """

        #array of unique labels
        unique = np.unique(y)

        #find the best node (will test for info_gain next)
        node = self.find_best_node(x, y, dep)

        #if all samples have the same label or there is no information to be 
        #gained by splitting
        if unique.size == 1 or node.info_gain == 0:
            l = max(node.dictionary, key = node.dictionary.get)
            return Node(label = l, dictionary = node.dictionary, 
                    info_gain = node.info_gain, depth = node.depth)
        else:
            #split the dataset into child datasets based on the attribute and split_point
            #stored in the best node
            children_datasets = self.split_dataset(x, y, node)

            for child_dataset in children_datasets:
                child_node = self.induce_decision_tree(child_dataset[0], child_dataset[1], dep + 1)
                node.add_child(child_node)

            return node

    def find_best_node(self, x, y, depth):
        """
        Returns the best node based on x: an NxK NumPy array representing N training
        instances of K attributes and y: an N-dim NumPy array containing the class label
        for each instance.
        """

        #initialise variables which store max info gain and the attribute and split point
        max_information_gain = 0
        attribute = None
        split_point = None

        #get the size of the array x
        row_size, column_size = x.shape

        #loop through each column of X
        for col in range(column_size):

            #sort attributes and labels based on column
            x, y = self.sort(x, y, col)

            #Find possible splitting points and, for each, their information gain
            #(can start from 1 as we need to split)
            for row in range(1,row_size):
                if x[row,col] != x[row-1,col]:
                    #slice the labels into two child sets
                    children = [child1, child2] = [y[:row], y[row:]]

                    #calculate information gain
                    info_gain = self.information_gain(y, children)

                    #test if this is max information gain so far
                    #if so, update holding values
                    if info_gain > max_information_gain:
                        max_information_gain = info_gain
                        attribute = col
                        split_point = x[row,col]

        #create a dictionary of unique labels and their counts to be stored in the node
        unique, counts = np.unique(y, return_counts = True)
        dictionary = dict(zip(unique, counts))

        #create node with attribute and value of max info_gain, the dictionary, and its depth
        node = Node(attribute, split_point, max_information_gain, dictionary, depth)

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
        Returns a list of the child sets (each child is a list of attributes and labels)
        """

        #Sort the attribute matrix and label matrix based on node.attribute
        x, y = self.sort(x, y, node.attribute)

        #get the row number in x where the data should be split 
        index = np.where(x[:,node.attribute] == node.split_point)
        sp = index[0][0]

        #Slice the datasets according to where the data should be split
        child1 = [x[:sp], y[:sp]]
        child2 = [x[sp:], y[sp:]]
        children = [child1, child2]

        return children

    def check_decision_tree(self, row, node):
        """
        Used to run a series of attributes (given by a list called row)
        through a decision tree starting at node.
        """

        if node.attribute != None:
            if row[node.attribute] < node.split_point:
                return self.check_decision_tree(row, node.child1)
            else:
                return self.check_decision_tree(row, node.child2)
        else:
            return node.label

    def print_decision_tree(self, node, max_depth=10):
        """
        Used to print a text-based visualisation of a decision tree.
        """
        if node.depth <= max_depth:
            if node.attribute != None:
                print("+---", "Attribute_" + str(node.attribute) + " < " + str(node.split_point), 
                    "(IG = " + str(round(node.info_gain, 4)) + " and Class Distribution = " 
                    + str(node.dictionary) + " and Depth = " + str(node.depth) + ")")
                if node.child1.depth < max_depth+1:
                    print(" "*4*(node.child1.depth), end = "")
                    self.print_decision_tree(node.child1)
                    print(" "*4*(node.child2.depth), end = "")
                    self.print_decision_tree(node.child2)
            else:
                print("+---", "Leaf", node.label, "(IG = " + str(round(node.info_gain, 4)) + 
                    " and Class Distribution = " + str(node.dictionary) + " and Depth = " + str(node.depth) + ")") 

        return self

        #######################################################################
        #                 ** TASK 4.1: COMPLETE THIS METHOD **
        #######################################################################

    def prune(self, x, y):
        """
        A function that is used to prune a decision tree.
        """
        self.prune_helper(self.node,x,y)
        return self

        #######################################################################
        #                 ** HELPER FUNCTIONS Q4 **
        #######################################################################

    def prune_helper(self, node, x, y):
        """
        Helper function to prune a decision tree.
        """

        if node.label != None:
            return True
        else:
            if self.prune_helper(node.child1, x, y) and self.prune_helper(node.child2, x, y):
                pre_accuracy = self.tree_accuracy(x,y)
                l = max(node.dictionary, key = node.dictionary.get)
                node.label = l

                children = [node.child1, node.child2]
                store = [node.attribute, node.split_point]
                node.child1 = None
                node.child2 = None
                node.attribute = None
                node.split_point = None

                post_accuracy = self.tree_accuracy(x,y)

                if post_accuracy >= pre_accuracy:
                    return True
                else:
                    node.child1 = children[0]
                    node.child2 = children[1]
                    node.label = None
                    node.attribute = store[0]
                    node.split_point = store[1]

        return False

    def tree_accuracy(self, x, y):
        predictions = self.predict(x)
        evaluator = eval.Evaluator()
        confusion = evaluator.confusion_matrix(predictions, y)
        return evaluator.accuracy(confusion)

class Node:
    """
    A Node object has the following data members:
        - the attribute and split point that results in the most informative split for
        some dataset (set to None if the Node is meant to contain only a label)
        - the information gain that is gained by splitting the dataset
        by the attribute and split point
        - a dictionary, where the keys are labels, and the values are
        the counts of these labels in an array
        - the depth of a node in a decision tree.
        - the label (set to None by default if the Node is meant to hold an attribute 
        and split_point)
        - the two child nodes that are pointed to by the parent (binary tree)

    It also has the add_child() method, which associates a child node with its parent
    """

    def __init__(self, attribute = None, split_point = None, info_gain = None, dictionary = None, depth = None, label = None, child1 = None, child2 = None):
        self.attribute = attribute
        self.split_point = split_point
        self.info_gain = info_gain
        self.dictionary = dictionary
        self.depth = depth
        self.label = label
        self.child1 = child1
        self.child2 = child2

    def add_child(self, child_node):
        """
        Adds a child node to its parent.
        """
        if self.child1 == None:
            self.child1 = child_node
        elif self.child2 == None:
            self.child2 = child_node
        else:
            print("Error adding child_node")

        




