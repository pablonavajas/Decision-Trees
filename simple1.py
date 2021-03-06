##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Train the model given in simple1.txt
# Prepared by: EPC
##############################################################################

import numpy as np
from classification import DecisionTreeClassifier
from eval import Evaluator
from test_class_exercise1 import Dataset

if __name__ == "__main__":
    print("Training " + "data/simple1.txt ...")

    print("Loading the training dataset")
   
    dataset = Dataset("data/simple1.txt")

    print("Training the decision tree ...");
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(dataset.attributes, dataset.labels)

    print("Visualisation of the Decision Tree ...")
    classifier.print_decision_tree(classifier.node)
