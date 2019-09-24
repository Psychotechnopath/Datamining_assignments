from sklearn.model_selection import StratifiedKFold, cross_validate
import openml as oml
from sklearn.neighbors import KNeighborsClassifier

SVHN = oml.datasets.get_dataset(41081)                                   #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_sub = X[0:9920]
y_sub = y[0:9920]

skfold = StratifiedKFold(3, shuffle=False, random_state=0)
knn = KNeighborsClassifier()
scores = cross_validate(knn, X_sub, y_sub, cv=skfold, scoring=['accuracy'], return_train_score=True)

print("Training accuracy of models {}".format(scores['train_accuracy']))
print("Standard deviation of training accuracies are {}".format([score.std() for score in scores['training_accuracy']]))
print("Testing accuracy of models {}".format(scores['test_accuracy']))
print("Standard deviation of Training accuracies {}".format([score.std() for score in scores['training_accuracy']]))
