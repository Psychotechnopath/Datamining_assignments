from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import openml as oml
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

SVHN = oml.datasets.get_dataset(41081)                                   #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_gray_std = np.load("std_grayscale_data.npy")

#Initialize default logistic regression
logisticReg = LogisticRegression()

#Initialize default knn classifier
knn = KNeighborsClassifier()

#Initialize default linear support vector machine
svc = LinearSVC()

X_gray_std_90_percent, X_gray_std_10_percent, y_90_percent, y_10_percent = train_test_split(X_gray_std, y, test_size=0.1, stratify=y, random_state=47)

# we provide a somewhat bad grid to illustrate the point:
param_grid = {'C': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]}
param_grid2 = {'n_neighbors': [5,15,25,35,45]}
# using the default scoring of accuracy:

#In GridSearchCV we use the default cv, as it implements Stratified K-folds cross-validation,
# From documentation: For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used.

grid = GridSearchCV(knn, param_grid=param_grid2, n_jobs=-1, cv=None, scoring=['accuracy'], return_train_score=True, refit='accuracy')
grid.fit(X_gray_std_10_percent, y_10_percent)
print("Grid-Search with accuracy")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))
print("Test set accuracy: {:.3f}".format(grid.score(X_gray_std_10_percent, y_10_percent)))

print("CV scores {}".format(grid.cv_results_))