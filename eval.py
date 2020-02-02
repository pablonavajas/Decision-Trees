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
            p[i] = confusion[i][i] / np.sum(confusion[0:, [i]])

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
            r[i] = confusion[i][i] / np.sum(confusion[[i], 0:])

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
            f[i] = 2 * p[i] * r[i] / (p[i] + r[i])

        macro_f = np.average(f)

        return (f, macro_f)


        #######################################################################
        #                 ** TASK 3.6: COMPLETE THIS METHOD **
        #######################################################################

class CrossValidator(object):

    # validate k_fold input from the user (private)
    def _k_fold_assert(self, k_folds, rows):
        assert (k_folds > 0 and k_folds <= rows), \
            "Number of labels in the predicted class not equal to number of labels in the ground truth class"

    # function to split the dataset (private)
    def _split_dataset(self, dataset, k_folds, random_fold):

        # get nr or rows of the dataset
        rows = len(dataset)

        # get length of one fold
        validation_set_len = rows // k_folds

        # get lower limit and upper limit for validation set rows
        split_row_1 = (validation_set_len) * random_fold
        split_row_2 = (validation_set_len) * (random_fold + 1)

        # extract validation set from dataset
        validation_set = dataset[split_row_1:split_row_2]

        # extract training set from dataset
        train_set = dataset[:split_row_1]
        train_set = np.append(train_set, dataset[split_row_2:], axis=0)

        return validation_set, train_set

    # run cross-validation
    def run(self, dataset, k_folds):
        # validate k_fold input from the user
        self._k_fold_assert(k_folds, len(dataset.attributes))

        # initialise an array for accuracies
        accuracy_array = np.empty(0);

        # randomly split the data into k subsets and get validation performance
        for fold in range(k_folds - 1):
            random_fold = random.randint(0, k_folds - 1) # 0-based

            validation_attributes, train_attributes = self._split_dataset(dataset.attributes, k_folds, random_fold)
            validation_labels, train_labels = self._split_dataset(dataset.labels, k_folds, random_fold)

            # training the decision tree
            classifier = DecisionTreeClassifier()
            classifier = classifier.train(train_attributes, train_labels)

            # get predictions based on the training
            predictions = classifier.predict(validation_attributes)

            # evaluation initialiser
            evaluator = Evaluator()
            classes = np.unique(dataset.labels);

            # build confusion matrix
            confusion = evaluator.confusion_matrix(predictions, validation_labels, classes)

            # get accuracy and append it to the array of accuracies
            accuracy = evaluator.accuracy(confusion)
            accuracy_array = np.append(accuracy, accuracy_array)

        #return accuracies_array, an average and a standard dev of those accuracies
        return accuracy_array, np.average(accuracy_array), np.std(accuracy_array)



