from sklearn.model_selection import cross_validate, train_test_split
import openml as oml
from sklearn.neighbors import KNeighborsClassifier

SVHN = oml.datasets.get_dataset(41081)                                                 #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_90_percent, X_10_percent, y_90_percent, y_10_percent = train_test_split(X, y, test_size=0.1, stratify=y, random_state=47)

print(len(X_10_percent))
print(len(X_10_percent))




# #Evaluate k-Nearest Neighbors, using default hyperparameter settings. Use cross-validation with 3 folds,
# #output the training accuracy and test accuracy including the standard deviations
#
# #Initialize default knn classifier
# knn = KNeighborsClassifier()
# #Use cross_validate with regular CV (As sample is already stratified, and we were not asked to use StratifiedKfold)
# scores = cross_validate(knn, X_10_percent, y_10_percent, cv=3, scoring=['accuracy'], return_train_score=True)
#
# #Report training accuracy + std, testing accuracy + std
# print("Training accuracy of models {}".format(scores['train_accuracy']))            #Training accuracy for 3 folds:
# print("Standard deviation of training accuracies are {}".format(scores['train_accuracy'].std())) #std over 3 folds:
# print("Testing accuracy of models {}".format(scores['test_accuracy']))               #Testing accuracy for 3 folds:
# print("Standard deviation of Training accuracies {}".format(scores['test_accuracy'].std()))      #std over 3 folds:
