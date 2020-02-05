##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Train the model given in train_full.txt
# Prepared by: EPC
##############################################################################

import numpy as np
from classification import DecisionTreeClassifier
from eval import Evaluator
from eval import CrossValidator
from test_class_exercise1 import Dataset

if __name__ == "__main__":
    print("Training " + "data/train_full.txt ...")

    print("Loading the training dataset")
   
    dataset = Dataset("data/train_full.txt")

    print("Training the decision tree ...");
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(dataset.attributes, dataset.labels)

    print("Visualisation of the Decision Tree ...")
    classifier.print_decision_tree(classifier.node)

    #######################################################################
    #               ** QUESTION 3.3: COMPLETE THIS METHOD **
    #######################################################################
    print("Performing cross-validation...")
    cv = CrossValidator()
    cross_validation_output = cv.run(dataset, 5)
    cv.print_evaluation_params(cross_validation_output[1])

    #######################################################################
    # ** QUESTION 3.4: EVALUATE MODEL TRAINED ON FULL DATASET  **
    # ** BEST ACCURACY MODEL FROM A SUBSET (CROSS-VALIDATED)   **
    #######################################################################
    print("\n\nQUESTION 3.4: BEST PERFORMING CLASSIFIER FROM CROSS-VAL vs CLASSIFIER TRAINED ON FULL DATASET")
    # get test labels
    test_dataset = Dataset("data/test.txt")

    # EVALUATION OF CLASSIFIER TRAINED ON FULL DATASET
    predictions = classifier.predict(test_dataset.attributes)

    # evaluation initializer
    evaluator = Evaluator()

    # build confusion matrix
    confusion = evaluator.confusion_matrix(predictions, test_dataset.labels)

    # get accuracy and append it to the array of accuracies
    print("\nEvaluation Parameters (using test.txt) of tree trained on full dataset ...")
    evaluator.print_four_eval_metrics(confusion)


    # EVALUATION ON THE BEST ACCURACY MODEL (FROM CROSS-VALIDATION)
    # choose the best classifier
    best_acc_cross_valid_classifier = cv.get_tree_with_max_accuracy(cross_validation_output)

    predictions = best_acc_cross_valid_classifier.predict(test_dataset.attributes)

    # evaluation initializer
    evaluator = Evaluator()

    # build confusion matrix
    confusion = evaluator.confusion_matrix(predictions, test_dataset.labels)

    # get accuracy and append it to the array of accuracies
    print("\nEvaluation Parameters on of the best performing tree on test.txt ...")
    evaluator.print_four_eval_metrics(confusion)

    #######################################################################
    #         ** QUESTION 3.5: COMBINE PREDICTIONS OF 10 TREES **
    #######################################################################
    print("\n\nQUESTION 3.5: COMBINED PREDICTIONS OF CLASSIFIERS TRAINED ON SUBSET OF DATA")

    # combine predictions from all 10 trees into one array called predictions
    predictions_combined = np.asarray(cross_validation_output[0][0].predict(test_dataset.attributes))
    for i in range(1, 5):
        predictions_combined = np.vstack((predictions, np.asarray(cross_validation_output[0][i].predict(test_dataset.attributes))))

    # get mode of those predictions across 10 trees
    predictions_combined_mode = cv.mode_2d(predictions_combined)

    print("\nEvaluation Parameters using combined predictions from 10 decision trees ...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions_combined_mode, test_dataset.labels)
    evaluator.print_four_eval_metrics(confusion)






