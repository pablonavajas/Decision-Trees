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
        print("Accuracy : %.3f" % self.accuracy(confusion))
        print("Precision: %.3f" % self.precision(confusion)[1])
        print("Recall   : %.3f" % self.recall(confusion)[1])
        print("F1 Score : %.3f" % self.f1_score(confusion)[1])

    def print_three_class_metrics(self, confusion):

        print("\nPrecision:")
        print(self.precision(confusion)[0])
        print("\nRecall   :")
        print(self.recall(confusion)[0])
        print("\nF1 Score :")
        print(self.f1_score(confusion)[0])




