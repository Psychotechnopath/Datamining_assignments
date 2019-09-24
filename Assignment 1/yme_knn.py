from sklearn.model_selection import cross_validate, train_test_split
import openml as oml
from sklearn.neighbors import KNeighborsClassifier

SVHN = oml.datasets.get_dataset(41081)                                                 #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_90_percent, X_10_percent, y_90_percent, y_10_percent = train_test_split(X, y, test_size=0.1, stratify=y)

knn = KNeighborsClassifier()
scores = cross_validate(knn, X_10_percent, y_10_percent, cv=3, scoring=['accuracy'], return_train_score=True)

print("Training accuracy of models {}".format(scores['train_accuracy']))
print("Standard deviation of training accuracies are {}".format(scores['train_accuracy'].std()))
print("Testing accuracy of models {}".format(scores['test_accuracy']))
print("Standard deviation of Training accuracies {}".format(scores['test_accuracy'].std()))
