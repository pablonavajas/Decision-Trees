##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Train the model given in train_full.txt
# Prepared by: PNH
##############################################################################

import numpy as np
from classification import DecisionTreeClassifier
from eval import Evaluator
from CrossValidator import CrossValidator
from test_class_exercise1 import Dataset

if __name__ == "__main__":

    #######################################################################
    #               ** QUESTION 3.1: COMPLETE THIS METHOD **              #
    #######################################################################

    print("\n\nQUESTION 3.1: Train Test.txt in three separate decision trees: ")

    test_dataset = Dataset("data/test.txt")

    test_labels = np.unique(test_dataset.labels)

    
    ##############################################
    #            TRAIN_FULL.TXT                  #
    ##############################################
    
    print("\n\nA. train_full.txt DECISION TREE:")
    
    print("\n\nLoading the training dataset...")
    full_dataset = Dataset("data/train_full.txt")

    print("Training the decision tree ...");
    full_classifier = DecisionTreeClassifier()
    full_classifier = full_classifier.train(full_dataset.attributes, full_dataset.labels)

    #print("Visualisation of the Decision Tree ...")
    #classifier.print_decision_tree(classifier.node)

    # EVALUATION OF CLASSIFIER TRAINED ON FULL DATASET
    full_predictions = full_classifier.predict(test_dataset.attributes)

    # evaluation initializer
    full_evaluator = Evaluator()

    # build confusion matrix
    full_confusion = full_evaluator.confusion_matrix(full_predictions, test_dataset.labels)
    print(full_confusion)

    print("\nEvaluation Parameters per class:")
    print(test_labels)
    full_evaluator.print_three_class_metrics(full_confusion)

    print("\nEvaluation Macro Parameters:")
    full_evaluator.print_four_eval_metrics(full_confusion)

    ##############################################
    #            TRAIN_SUB.TXT                   #
    ##############################################
    
    print("\n\nB. train_sub.txt DECISION TREE:")
    
    print("\n\nLoading the training dataset...")
    sub_dataset = Dataset("data/train_sub.txt")

    print("Training the decision tree ...");
    sub_classifier = DecisionTreeClassifier()
    sub_classifier = sub_classifier.train(sub_dataset.attributes, sub_dataset.labels)

    #print("Visualisation of the Decision Tree ...")
    #classifier.print_decision_tree(sub_classifier.node)

    # EVALUATION OF CLASSIFIER TRAINED ON FULL DATASET
    sub_predictions = sub_classifier.predict(test_dataset.attributes)

    # evaluation initializer
    sub_evaluator = Evaluator()

    # build confusion matrix
    sub_confusion = sub_evaluator.confusion_matrix(sub_predictions, test_dataset.labels)
    print(sub_confusion)

    print("\nEvaluation Parameters per class:")
    print(test_labels)
    sub_evaluator.print_three_class_metrics(sub_confusion)

    print("\nEvaluation Macro Parameters:")
    sub_evaluator.print_four_eval_metrics(sub_confusion)

    
    ##############################################
    #            TRAIN_NOISY.TXT                 #
    ##############################################
    
    print("\n\nC. train_noisy.txt DECISION TREE:")
    
    print("\n\nLoading the training dataset...")
    noisy_dataset = Dataset("data/train_noisy.txt")

    print("Training the decision tree ...");
    noisy_classifier = DecisionTreeClassifier()
    noisy_classifier = noisy_classifier.train(noisy_dataset.attributes, noisy_dataset.labels)

    #print("Visualisation of the Decision Tree ...")
    #classifier.print_decision_tree(sub_classifier.node)

    # EVALUATION OF CLASSIFIER TRAINED ON FULL DATASET
    noisy_predictions = noisy_classifier.predict(test_dataset.attributes)

    # evaluation initializer
    noisy_evaluator = Evaluator()

    # build confusion matrix
    noisy_confusion = noisy_evaluator.confusion_matrix(noisy_predictions, test_dataset.labels)
    print(noisy_confusion)

    print("\nEvaluation Parameters per class:")
    print(test_labels)
    noisy_evaluator.print_three_class_metrics(noisy_confusion)
    
    print("\nEvaluation Macro Parameters:")
    noisy_evaluator.print_four_eval_metrics(noisy_confusion)

