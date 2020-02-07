import numpy as np
from classification import DecisionTreeClassifier
from eval import Evaluator


class CrossValidator(object):
    """
    CrossValidator class contains the following:
        Main data members:
            - split_dataset (which splits the dataset into k folds)
            - run which runs the cross-validation and returns all k
              decision trees and their metrics (also reports average,
              standard deviation and max of each one of the metrics)
        Helper functions:
            - run_evaluation which performs evaluation on one classifier
              and returns the classifier and it's evaluation metrics
            - k_fold_assert which checks for a valid k_folds number
            - print_evaluation_params to print out a table with the
              evaluation parameters for 10 trees to command line
            - mode2d which gets the mode for columns of an array
            - get_tree_with_max_accuracy which takes the output
              from the run function and returns a tree with maximum
              accuracy
    """

    #######################################################################
    #                 ** TASK 3.6: COMPLETE THIS METHOD **
    #######################################################################

    def split_dataset(self, dataset, k_folds, validation_fold):
        """
        splits the dataset into k folds and returns
        the resulting validation set and training set
        """

        # get nr or rows of the dataset
        rows = len(dataset)

        # get length of one fold
        validation_set_len = rows // k_folds

        # get lower limit and upper limit for validation fold rows
        split_row_1 = validation_set_len * validation_fold
        split_row_2 = validation_set_len * (validation_fold + 1)

        # extract validation set from dataset
        validation_set = dataset[split_row_1:split_row_2]

        # extract training set from dataset
        train_set = dataset[:split_row_1]
        train_set = np.append(train_set, dataset[split_row_2:], axis=0)

        return validation_set, train_set

    def run(self, dataset, k_folds):
        """
        performs cross-validation on the dataset using k_folds
        returns k number of classifiers and an array with the evaluation
        metrics for the k number of classifiers
        evaluation metrics include accuracy, precision, recall and f1 score
        as well as averages, standard deviations and maximum value for those
        metrics
        """

        # validate k_fold input from the user
        self.k_fold_assert(k_folds, len(dataset.attributes))

        # initialise an array for accuracies
        eval_data_frame = np.empty(0)
        decision_trees = np.empty(0)

        # shuffle the dataset (randomise the split)
        s = np.arange(dataset.labels.shape[0])
        np.random.shuffle(s)
        attributes = dataset.attributes[s]
        labels = dataset.labels[s]

        # randomly split the data into k subsets and get validation performance
        for fold in range(k_folds):

            validation_attributes, train_attributes = self.split_dataset(
                attributes, k_folds, fold)
            validation_labels, train_labels = self.split_dataset(labels,
                                                                 k_folds, fold)

            classifier, eval_params = self.run_evaluation(validation_attributes,
                                                          validation_labels,
                                                          train_attributes,
                                                          train_labels,
                                                          np.unique(labels))

            if fold == 0:
                eval_data_frame = eval_params
                decision_trees = classifier
            else:
                eval_data_frame = np.append(eval_data_frame, eval_params,
                                            axis=0)
                decision_trees = np.append(decision_trees, classifier)

        av = np.mean(eval_data_frame, axis=0)
        std = np.std(eval_data_frame, axis=0)
        max_val = np.max(eval_data_frame, axis=0)

        eval_data_frame = np.append(eval_data_frame, av)
        eval_data_frame = np.append(eval_data_frame, std)
        eval_data_frame = np.append(eval_data_frame, max_val)

        return decision_trees, eval_data_frame

    #######################################################################
    #                         ** HELPER FUNCTIONS **
    #######################################################################

    def run_evaluation(self, validation_attributes, validation_labels,
                       train_attributes, train_labels, unique_labels):
        """
        runs the evaluation and returns the classifier and evaluation
        parameters back
        """

        # training the decision tree
        classifier = DecisionTreeClassifier()
        classifier = classifier.train(train_attributes, train_labels)

        # get predictions based on the training
        predictions = classifier.predict(validation_attributes)

        # evaluation initialiser
        evaluator = Evaluator()

        # build confusion matrix
        confusion = evaluator.confusion_matrix(predictions, validation_labels,
                                               unique_labels)

        # get accuracy and append it to the array of accuracies
        accuracy = evaluator.accuracy(confusion)
        p, macro_p = evaluator.precision(confusion)
        r, macro_r = evaluator.recall(confusion)
        f, macro_f = evaluator.f1_score(confusion)
        eval_params = np.array([[accuracy, macro_p, macro_r, macro_f]],
                               dtype=float)

        return classifier, eval_params

    def k_fold_assert(self, k_folds, rows):
        """ validate k_fold input from the user (private) """
        assert (k_folds > 0 and k_folds <= rows), \
            "Invalid number of k_folds"

    def print_evaluation_params(self, eval_data_frame):
        """ prints the evaluation parameters in a table to command line """
        eval_data_frame = eval_data_frame.reshape(-1, 4)

        print("                   Accuracy    Precision    Recall    F1 ")
        nr_of_rows = eval_data_frame.shape[0]
        for row in range(nr_of_rows):
            if row < nr_of_rows - 3:
                row_label = "Decision Tree " + str(row)
            elif row < nr_of_rows - 2:
                row_label = "Average        "
            elif row < nr_of_rows - 1:
                row_label = "St. Dev.       "
            else:
                row_label = "Max Value      "

            print(" %s   %.3f     %.3f      %.3f   %.3f" % (
                row_label, eval_data_frame[row][0],
                eval_data_frame[row][1], eval_data_frame[row][2],
                eval_data_frame[row][3]))

    def mode_2d(self, array_n_x_k):
        """ get an array of mode values for all columns of array (n x k) """
        k = array_n_x_k.shape[1]
        mode_array = np.empty(0)
        for i in range(k):
            value, count = np.unique(array_n_x_k[:, i], return_counts=True)
            indices = (-count).argsort()  # minus for descending order
            value = value[indices]  # rearrange in descending order of count
            mode_array = np.append(mode_array, value[0])
        return mode_array

    def get_tree_with_max_accuracy(self, classifiers_and_eval_params):
        classifiers, eval_data_frame = classifiers_and_eval_params
        eval_data_frame = eval_data_frame.reshape(-1, 4)

        nr_trees = len(eval_data_frame) - 3
        accuracy_array = eval_data_frame[:nr_trees, 0]
        max_accuracy_index = np.argmax(accuracy_array)
        return classifiers[max_accuracy_index]
