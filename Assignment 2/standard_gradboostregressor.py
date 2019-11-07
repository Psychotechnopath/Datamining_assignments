import numpy as np
np.random.seed(47)
from sklearn.ensemble import GradientBoostingRegressor as GBR


lr = np.logspace(-4, 0, 5)                                       #Create a range of parameter learning_rate, on a logarithmic scale, from 1e-4 to 1e-1
max_depth = np.linspace(1,5, 1)                                  #Create a range of parameter max_depth, on a linspace scale, from 1 to 5
sample_learning_rate = np.random.choice(lr, 3, replace=False)    #Sample 3 random points out of the parameter c range
sample_max_depth = np.random.choice(max_depth, 3, replace=False) #Sample 3 random points out of the parameter gamma range


GBR_hyperparams = np.array(list(zip(sample_learning_rate, sample_max_depth)))  #Zip together learning_rate and max_depth, cast it to the right data structure (np.array)
learning_rate_max_depth_fixed = np.array([(l, 1) for l in lr])   #Create a np.array where learning_rate varies and max_depth is fixed
max_depth_learning_rate_fixed = np.array([(0, m) for m in max_depth])       #Create a np.array where max_depth varies and learning_rate is fixed

gbr = GBR(n_estimators=1000, max_depth=0, learning_rate=0)

