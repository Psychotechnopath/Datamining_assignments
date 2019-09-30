from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import openml as oml
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle


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


param_grid_course = {'C': [1e-12, 1e-6, 1e-4, 1e-2, 1e-1, 1, 1e1, 1e2, 1e4, 1e6, 1e12]}
param_grid2 = {'n_neighbors': [5,15,25,35,45]}
# using the default scoring of accuracy:

#In GridSearchCV we use the default cv, as it implements Stratified K-folds cross-validation with three folds,
# From documentation: For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used.
grid = GridSearchCV(knn, param_grid=param_grid2, n_jobs=-1, cv=None, scoring=['accuracy'], return_train_score=True, refit='accuracy')
grid.fit(X_gray_std_10_percent, y_10_percent)
print("Grid-Search with accuracy")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))
print("CV scores {}".format(grid.cv_results_))

cvresults_forfile = grid.cv_results_
f = open("cvresults_file.pkl","wb")
pickle.dump(cvresults_forfile,f)
f.close()

# print(grid.cv_results_['params'])
# print(grid.cv_results_['params'])
#
# print(grid.cv_results_['split1_train_accuracy'])
# print(type(grid.cv_results_['split1_train_accuracy']))
#
# plt.subplot(2,1,2)
# plt.title('Training and testing accuracies for KNN in a Grid Search')
# plt.plot(grid.cv_results_['param_n_neighbors'], grid.cv_results_['split0_train_accuracy'], '-o')
# plt.plot(grid.cv_results_['param_n_neighbors'], grid.cv_results_['split1_train_accuracy'], '-o')
# plt.plot(grid.cv_results_['param_n_neighbors'], grid.cv_results_['split2_train_accuracy'], '-o')
# plt.legend(['Fold 1', 'Fold 2', 'Fold 3'], loc='upper right')
# plt.xlabel("Number of neighbors")
# plt.ylabel("Training Accuracy")
#
# plt.plot(grid.cv_results_['param_n_neighbors'], grid.cv_results_['split0_test_accuracy'], '-o')
# plt.plot(grid.cv_results_['param_n_neighbors'], grid.cv_results_['split1_test_accuracy'], '-o')
# plt.plot(grid.cv_results_['param_n_neighbors'], grid.cv_results_['split2_test_accuracy'], '-o')
# plt.legend(['Fold 1', 'Fold 2', 'Fold 3'], loc='upper right')
# plt.xlabel("Number of neighbors")
# plt.ylabel("Testing Accuracy")
# plt.show()

