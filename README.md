## CO395 Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.

### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class. Your task 
is to implement the ``train()`` and ``predict()`` methods.


- ``eval.py``

	* Contains the skeleton code for the ``Evaluator`` class. Your task is to 
implement the ``confusion_matrix()``, ``accuracy()``, ``precision()``, 
``recall()``, and ``f1_score()`` methods.


- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods defined in ``classification.py`` and ``eval.py``.


### Instructions

#### Importing the datasets
File ``readData.py`` has been created to import the data from textual format into some numpy
arrays containing labels and attributes, to be used for the training of the decision trees.


#### Decision trees training and printing
To perform the training and print out decision trees for all the datasets provided in 
the ``\data`` directory, python files have been created with 
filenames corresponding to the filename of the datasets
(i.e. ``toy.py`` is the file to be ran to print out decision tree trained on ``toy.txt`` dataset).
So overall there is ``toy.py``, ``simple1.py``, ``simple2.py`` and ``train_full.py`` which have
been created to run the training and printing out of the decision trees on the corresponding datasets.

#### Evaluation
For evaluation part, file ``evaluation_q3.py`` should be executed to print out all the results
for part 3 of the coursework. This file utilises the ``CrossValidator`` class to perform
cross-validation for the latter section of part 3 of this coursework. The ``CrossValidator``
class is located in a separate file called ``CrossValidator.py``.

#### Pruning 
For the pruning part of the coursework, file ``pruning_q4.py`` should be executed, which will
print out all the results for this part. This utilises the ``prune()`` method from inside the ``classification.py``
file.




    





