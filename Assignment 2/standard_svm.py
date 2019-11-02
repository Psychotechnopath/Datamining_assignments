import openml as oml
import numpy as np
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate
import pickle

# Download Wall Robot Navigation data from OpenML.
robotnav = oml.datasets.get_dataset(1497)
X_nav, y_nav, cats, attrs = robotnav.get_data(dataset_format='array',
target=robotnav.default_target_attribute)

#Download Wall robot arm Data from OpenMl
robotarm = oml.datasets.get_dataset(189)
Xr, yr, catsr, attrsr = robotarm.get_data(dataset_format='dataframe',
target=robotarm.default_target_attribute)

c_points = np.logspace(-12,12, 25)
gamma_points = np.logspace(-12, 12, 25)
sample_c = np.random.choice(c_points, 10 , replace=False)
sample_g = np.random.choice(gamma_points, 10, replace=False)
random_hyperparams = zip(sample_c, sample_g)

score_list = []

for c_param, g_param in random_hyperparams:
    svc = SVC(C=c_param, kernel='rbf', gamma=g_param)
    score = cross_validate(svc, X_nav, y_nav, cv=3, scoring=['mse'])
    score_list.append(score)

with open("score_list.pkl", "wb") as f:
    pickle.dump(score_list, f)






