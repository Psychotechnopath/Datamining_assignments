import pickle
import numpy as np

results_file = open('C:/Users/Gebruiker/MEGA/Master DSE/Data Mining/Assignments/Assignment 1/cache/logreg_pca_score60factors.pkl','rb')
cv_results_ = pickle.load(results_file)
print(np.mean(cv_results_['test_acc']))
results_file.close()