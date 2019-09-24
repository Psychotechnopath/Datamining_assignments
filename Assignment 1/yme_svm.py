from sklearn.model_selection import train_test_split, cross_validate
import openml as oml
from sklearn.svm import LinearSVC

SVHN = oml.datasets.get_dataset(41081)                                   #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_90_percent, X_10_percent, y_90_percent, y_10_percent = train_test_split(X, y, test_size=0.1, stratify=y, random_state=47)
svc = LinearSVC()
scores = cross_validate(svc, X_10_percent, y_10_percent, cv=3, scoring=['accuracy'], return_train_score=True)



#Report training accuracy + std, testing accuracy + std
print("Training accuracy of models {}".format(scores['train_accuracy']))            #Training accuracy for 3 folds:
print("Standard deviation of training accuracies are {}".format(scores['train_accuracy'].std())) #std over 3 folds:
print("Testing accuracy of models {}".format(scores['test_accuracy']))               #Testing accuracy for 3 folds:
print("Standard deviation of Training accuracies {}".format(scores['test_accuracy'].std()))      #std over 3 folds:
