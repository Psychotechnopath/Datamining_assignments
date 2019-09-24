from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import openml as oml
from sklearn.neighbors import KNeighborsClassifier

SVHN = oml.datasets.get_dataset(41081)                                                 #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_90_percent, X_10_percent, y_90_percent, y_10_percent = train_test_split(X, y, test_size=0.1, stratify=y)   #Use train_test_split in a "clever" way to generate 10% stratified sub-sample


skfold = StratifiedKFold(3, shuffle=False, random_state=0)
knn = KNeighborsClassifier()
scores = cross_validate(knn, X_sub, y_sub, cv=skfold, scoring=['accuracy'], return_train_score=True)
#
# x_list = []
# y_list = []
#
#
# for train_index, test_index in skfold.split(X,y):
#     x_list.append((X[train_index], X[test_index]))
#     y_list.append((y[train_index], y[test_index]))
#
# print(len(x_list))
# print(len(y_list))
# print(type(y_list[0]))


#
# print("Training accuracy of models {}".format(scores['train_accuracy']))
# print("Standard deviation of training accuracies are {}".format([score.std() for score in scores['training_accuracy']]))
# print("Testing accuracy of models {}".format(scores['test_accuracy']))
# print("Standard deviation of Training accuracies {}".format([score.std() for score in scores['training_accuracy']]))
