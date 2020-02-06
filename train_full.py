##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Train the model given in train_full.txt
# Prepared by: EPC
##############################################################################

import numpy as np
from classification import DecisionTreeClassifier
from eval import Evaluator
from CrossValidator import CrossValidator
from readData import Dataset
import sys

if __name__ == "__main__":
    print("Training " + "data/train_full.txt ...")
    #print("Training " + sys.argv[1])

    print("Loading the training dataset")
   
    dataset = Dataset("data/train_full.txt")
    #dataset = Dataset(sys.argv[1])
    

    print("Training the decision tree ...");
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(dataset.attributes, dataset.labels)

    print("Visualisation of the Decision Tree ...")
    classifier.print_decision_tree(classifier.node)



