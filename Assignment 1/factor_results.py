import pickle
import numpy as np

f = open('C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 1/saved_files/logreg_pca_score320factors.pkl', "rb")
cv_results = pickle.load(f)
print(np.mean(cv_results['test_acc'])) #knn 20 factors: 0.19378

print(cv_results)