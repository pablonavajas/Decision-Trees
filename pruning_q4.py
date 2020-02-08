from classification import DecisionTreeClassifier
from eval import Evaluator
from readData import Dataset

validation = Dataset("data/validation.txt")
test = Dataset("data/test.txt")

#######################################################################
#         ** QUESTION 4.1: PRUNING on full and noisy datasets**       #
#######################################################################
print("====================================================================")
print("========================= QUESTION 4.1 =============================")
print("====================================================================")

# print(" \n====== TESTING OF train_full.txt CLASSIFIER on validation.txt
# ======")

#######################################################################
#           ** train_full.txt model on test.txt **
#######################################################################
print(" \n====== TESTING OF train_full.txt CLASSIFIER on validation.txt ======")

# open train dataset
train_full = Dataset("data/train_full.txt")
classifier_full = DecisionTreeClassifier()
classifier_full = classifier_full.train(train_full.attributes,
                                        train_full.labels)

evaluator = Evaluator()
print("\nEvaluation of train_full.txt on test.txt BEFORE pruning: ")
predictions = classifier_full.predict(test.attributes)
confusion = evaluator.confusion_matrix(predictions, test.labels)
evaluator.print_four_eval_metrics(confusion)

tree_max_depth = classifier_full.max_depth

print("\nPruning the Tree...")
pruned_classifier_full = classifier_full.prune(validation.attributes,
                                               validation.labels)

pruned_tree_max_depth = pruned_classifier_full.max_depth

print("\nEvaluation of train_full.txt model on test.txt AFTER pruning: ")
predictions = pruned_classifier_full.predict(test.attributes)
confusion = evaluator.confusion_matrix(predictions, test.labels)
evaluator.print_four_eval_metrics(confusion)

#######################################################################
#           ** train_noisy.txt model on test.txt **
#######################################################################
print(" \n====== TESTING OF train_noisy.txt CLASSIFIER on test.txt ======")

# open train dataset
train_noisy = Dataset("data/train_noisy.txt")

classifier_noisy = DecisionTreeClassifier()
classifier_noisy = classifier_noisy.train(train_noisy.attributes,
                                          train_noisy.labels)

noisy_tree_max_depth = classifier_noisy.max_depth

print("\nEvaluation of train_noisy.txt on test.txt BEFORE pruning: ")
predictions = classifier_noisy.predict(test.attributes)
evaluator = Evaluator()
confusion = evaluator.confusion_matrix(predictions, test.labels)
evaluator.print_four_eval_metrics(confusion)

print("\nPruning the Tree...")
pruned_classifier_noisy = classifier_noisy.prune(validation.attributes,
                                                 validation.labels)

pruned_noisy_max_depth = pruned_classifier_noisy.max_depth

print("\nEvaluation of train_noisy.txt model on test.txt AFTER pruning: ")
predictions = pruned_classifier_noisy.predict(test.attributes)
confusion = evaluator.confusion_matrix(predictions, test.labels)
evaluator.print_four_eval_metrics(confusion)

print("\n\n")

#######################################################################
#                       ** QUESTION 4.2: **                           #
#######################################################################
print("====================================================================")
print("========================= QUESTION 4.2 =============================")
print("====================================================================")

print("Model trained using \"train_full.txt\"")
print("The maximal depth of the un-pruned tree is:")
print(tree_max_depth)

print("\nThe maximal depth of the pruned tree is:")
print(pruned_tree_max_depth)

print("\n")

print("Model trained using \"train_noisy.txt\"")
print("The maximal depth of the un-pruned tree is:")
print(noisy_tree_max_depth)

print("The maximal depth of the pruned tree is:")
print(pruned_noisy_max_depth)

print("\n")
