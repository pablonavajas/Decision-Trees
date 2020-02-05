##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np
from classification import DecisionTreeClassifier
import random

class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if class_labels is None:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################

        # validate inputs
        assert len(prediction) == len(annotation), \
            "Number of labels in the predicted class not equal to number of labels in the ground truth class"

        results = np.vstack((annotation, prediction)).T

        i = 0

        while i < len(results):
            idx = np.where(class_labels == results[i][0])
            idx2 = np.where(class_labels == results[i][1])

            idx_int = int(idx[0])
            idx2_int = int(idx2[0])

            # print('The test ' + str(i) + ' had label ' + results[i][0] + ' and prediction ' + results[i][1])
            # print('Hence, it will be stored in row ' + str(idx_int) + ' col ' + str(idx2_int))

            confusion[idx_int][idx2_int] += 1
            i += 1


        return confusion

    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################
        n = len(confusion)
        accuracy = sum(confusion[i][i] for i in range(n)) / np.sum(confusion)

        return accuracy

    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################
        n = len(confusion)
        for i in range(n):
            denominator = np.sum(confusion[0:, [i]])
            if(denominator):
                p[i] = confusion[i][i] / denominator
            else:
                p[i] = 0
        # You will also need to change this
        macro_p = np.average(p)

        return (p, macro_p)

    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################

        # You will also need to change this
        n = len(confusion)
        for i in range(n):
            denominator = np.sum(confusion[[i], 0:])
            if (denominator):
                r[i] = confusion[i][i] / denominator
            else:
                r[i] = 0

        # You will also need to change this
        macro_r = np.average(r)

        return (r, macro_r)

    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion),))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        (p, macro_p) = self.precision(confusion)
        (r, macro_r) = self.recall(confusion)

        n = len(confusion)
        for i in range(n):
            denominator = p[i] + r[i]
            if (denominator):
                f[i] = 2 * p[i] * r[i] / denominator
            else:
                f[i] = 0


        macro_f = np.average(f)

        return (f, macro_f)

    def print_four_eval_metrics(self, confusion):
        print("Accuracy : %.5f" % self.accuracy(confusion))
        print("Precision: %.5f" % self.precision(confusion)[1])
        print("Recall   : %.5f" % self.recall(confusion)[1])
        print("F1 Score : %.5f" % self.f1_score(confusion)[1])



#######################################################################
#                 ** TASK 3.6: COMPLETE THIS METHOD **
#######################################################################

class CrossValidator(object):

    # validate k_fold input from the user (private)
    def _k_fold_assert(self, k_folds, rows):
        assert (k_folds > 0 and k_folds <= rows), \
            "Number of labels in the predicted class not equal to number of labels in the ground truth class"

    # function to split the dataset (private)
    def _split_dataset(self, dataset, k_folds, validation_fold):

        # get nr or rows of the dataset
        rows = len(dataset)

        # get length of one fold
        validation_set_len = rows // k_folds

        # get lower limit and upper limit for validation set rows
        split_row_1 = validation_set_len * validation_fold
        split_row_2 = validation_set_len * (validation_fold + 1)

        # extract validation set from dataset
        validation_set = dataset[split_row_1:split_row_2]

        # extract training set from dataset
        train_set = dataset[:split_row_1]
        train_set = np.append(train_set, dataset[split_row_2:], axis=0)

        return validation_set, train_set

    def run_evaluation(self, validation_attributes, validation_labels, train_attributes, train_labels, unique_labels):

        # training the decision tree
        classifier = DecisionTreeClassifier()
        classifier = classifier.train(train_attributes, train_labels)

        # get predictions based on the training
        predictions = classifier.predict(validation_attributes)

        # evaluation initialiser
        evaluator = Evaluator()
        classes = np.unique(unique_labels);

        # build confusion matrix
        confusion = evaluator.confusion_matrix(predictions, validation_labels, classes)

        # get accuracy and append it to the array of accuracies
        accuracy = evaluator.accuracy(confusion)
        p, macro_p = evaluator.precision(confusion)
        r, macro_r = evaluator.recall(confusion)
        f, macro_f = evaluator.f1_score(confusion)
        eval_params = np.array([[accuracy, macro_p, macro_r, macro_f]], dtype=float)

        return classifier, eval_params

    # run cross-validation
    def run(self, dataset, k_folds):
        # validate k_fold input from the user
        self._k_fold_assert(k_folds, len(dataset.attributes))

        # initialise an array for accuracies
        eval_data_frame = np.empty(0);
        decision_trees = np.empty(0)#, type=DecisionTreeClassifier);


        # shuffle the dataset
        s = np.arange(dataset.labels.shape[0])
        np.random.shuffle(s)
        attributes = dataset.attributes[s]
        labels = dataset.labels[s]

        # randomly split the data into k subsets and get validation performance
        for fold in range(k_folds):

            validation_attributes, train_attributes = self._split_dataset(attributes, k_folds, fold)
            validation_labels, train_labels = self._split_dataset(labels, k_folds, fold)

            classifier, eval_params = self.run_evaluation(validation_attributes, validation_labels,
                                                          train_attributes, train_labels, np.unique(labels))

            if fold == 0:
                eval_data_frame = eval_params
                decision_trees = classifier
            else:
                eval_data_frame = np.append(eval_data_frame, eval_params, axis=0)
                decision_trees = np.append(decision_trees, classifier)

        av = np.mean(eval_data_frame, axis=0)
        std = np.std(eval_data_frame, axis=0)
        max_val = np.max(eval_data_frame, axis=0)

        eval_data_frame = np.append(eval_data_frame, av)
        eval_data_frame = np.append(eval_data_frame, std)
        eval_data_frame = np.append(eval_data_frame, max_val)

        return decision_trees, eval_data_frame

    def print_evaluation_params(self, eval_data_frame):
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

            print(" %s   %.5f     %.5f      %.5f   %.5f" % (row_label, eval_data_frame[row][0],
                     eval_data_frame[row][1], eval_data_frame[row][2], eval_data_frame[row][3]))

    #get an array of mode for all columns of array (n x k)
    def mode_2d(self, array_n_x_k):
        k = array_n_x_k.shape[1]
        mode_array = np.empty(0)
        for i in range(k):
            value, count = np.unique(array_n_x_k[:, i], return_counts=True)
            indices = (-count).argsort() #minus for descending order
            value = value[indices]       #rearrange in descending order of count
            mode_array = np.append(mode_array, value[0])
        return mode_array


    def get_tree_with_max_accuracy(self, classifiers_and_eval_params):
        classifiers, eval_data_frame = classifiers_and_eval_params
        eval_data_frame = eval_data_frame.reshape(-1, 4)

        nr_trees = len(eval_data_frame) - 3
        accuracy_array = eval_data_frame[:nr_trees, 0]
        max_accuracy_index = np.argmax(accuracy_array)
        return classifiers[max_accuracy_index]
