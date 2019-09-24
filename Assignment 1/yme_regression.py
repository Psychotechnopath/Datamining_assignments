from sklearn.model_selection import StratifiedKFold, cross_validate
import openml as oml
from sklearn.linear_model import LogisticRegression


SVHN = oml.datasets.get_dataset(41081)                                   #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)


skfold = StratifiedKFold(3, shuffle=False, random_state=0)

logisticReg = LogisticRegression()
scores = cross_validate(logisticReg, X_sub, y_sub, cv=skfold, scoring=['accuracy'], return_train_score=True)

print("Training accuracy of models {}".format(scores['train_accuracy']))
print("Standard deviation of training accuracies are {}".format(scores['train_accuracy'].std()))
print("Testing accuracy of models {}".format(scores['test_accuracy']))
print("Standard deviation of Training accuracies {}".format(scores['test_accuracy'].std()))
