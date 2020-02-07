##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Train the model given in train_full.txt
# Prepared by: EPC
##############################################################################

from classification import DecisionTreeClassifier
from readData import Dataset

if __name__ == "__main__":
    print("Training " + "data/train_full.txt ...")

    print("Loading the training dataset")

    dataset = Dataset("data/train_full.txt")

    print("Training the decision tree ...");
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(dataset.attributes, dataset.labels)

    print("Visualisation of the Decision Tree ...")
    classifier.print_decision_tree(classifier.node, max_depth=4)
