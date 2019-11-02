import pickle
from pbreg import ProbabilisticRandomForestRegressor
from pbreg import EI
import numpy as np
from matplotlib import pyplot as plt
import openml as oml

robotnav = oml.datasets.get_dataset(1497)
X_nav, y_nav, cats, attrs = robotnav.get_data(dataset_format='array',
target=robotnav.default_target_attribute)

c_points = np.logspace(-12,12, 25)                           #Create a range of parameter c, on a logarithmic scale, from 1e-12 to 1e12
gamma_points = np.logspace(-12, 12, 25)                      #Create a range of parameter gamma, on a logarithmic scale, from 1e-12 to 1e12
sample_c = np.random.choice(c_points, 10 , replace=False)    #Sample 10 random points out of the parameter c range
sample_g = np.random.choice(gamma_points, 10, replace=False) #Sample 10 random points out of the parameter gamma range
random_hyperparams = np.array(list(zip(sample_c, sample_g))) #Zip together sample_c and sample_g, cast it to the right data structure (np.array)
c_points_gamma_fixed = np.array([(1, c) for c in c_points])  #Create a np.array where c varies and gamma is fixed

with open('score_list.pkl', 'rb') as score_list:
    unpickle_score_list = pickle.load(score_list)

accuracy_means = np.array([1 - np.mean(i['test_accuracy']) for i in unpickle_score_list]) #Calculate the means out of the returned accuracy from the standard svm

rf = ProbabilisticRandomForestRegressor(n_estimators=100)
rf.fit(random_hyperparams, accuracy_means)
Y_pred, sigma = rf.predict(c_points_gamma_fixed, return_std=True)
print(Y_pred)
print(sigma)
print(c_points)
# Plot surrogate model

plt.ylabel('Surrogate model error')
plt.xscale('log')
plt.plot(c_points, Y_pred, label=u'Prediction')
plt.fill_between(c_points.ravel(),Y_pred-2*sigma,Y_pred+2*sigma,alpha=0.1,label='Uncertainty')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Plot acquisition function
# plt.subplot(2, 1, 2)
# plt.xlabel('Hyperparameter value')
# plt.ylabel('Expected Improvement')
# plt.plot(c, EI(rf,Y_pred))
# plt.show()