from sklearn.model_selection import train_test_split, cross_validate
import openml as oml
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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

for i in [20,40,60,80,160,320]:
    pipeline= Pipeline([("PCA", PCA(n_components=i, svd_solver='auto')), ("knn", knn)])
    print(pipeline.named_steps["PCA"])

    X_gray_std_90_percent, X_gray_std_10_percent, y_90_percent, y_10_percent = train_test_split(X_gray_std, y, test_size=0.1, stratify=y, random_state=47)

    #Use cross_validate with regular CV (As sample is already stratified, and we were not asked to use StratifiedKfold)
    scores = cross_validate(pipeline, X_gray_std_10_percent, y_10_percent, cv=3, scoring=['accuracy'], return_train_score=True)
    #Report training accuracy + std, testing accuracy + std
    print("Training accuracy of models {}".format(scores['train_accuracy']))            #Training accuracy for 3 folds:
    print("Standard deviation of training accuracies are {}".format(scores['train_accuracy'].std())) #std over 3 folds:
    print("Testing accuracy of models {}".format(scores['test_accuracy']))               #Testing accuracy for 3 folds:
    print("Standard deviation of Training accuracies {}".format(scores['test_accuracy'].std()))      #std over 3 folds:
    results_dict = {'train_acc': scores['train_accuracy'], 'train_std' : scores['train_accuracy'].std(), 'test_acc': scores['test_accuracy'], 'test_std':scores['test_accuracy'].std()}
    f = open("C:/Users/Gebruiker/MEGA/Master DSE/Data Mining/Assignments/Assignment 1/cache/knn_pca_score{}factors.pkl".format(i), "wb")
    pickle.dump(results_dict, f)
    f.close()
