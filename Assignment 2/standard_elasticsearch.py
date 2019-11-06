from sklearn.linear_model import ElasticNet
import numpy as np

alpha_points = np.logspace(-12,12, 25)                                      #Create a range of parameter c, on a logarithmic scale, from 1e-12 to 1e12
l1_ratio = np.linspace(0,1)
sample_alpha = np.random.choice(alpha_points, 10, replace=False)
sample_l1 = np.random.choice(l1_ratio, 10, replace=False)
random_hyperparams = np.array(list(zip(sample_alpha, sample_l1)))  #Zip together learning_rate and max_depth, cast it to the right data structure (np.array)
learning_rate_max_depth_fixed = np.array([(l, 1) for l in sample_alpha])   #Create a np.array where learning_rate varies and max_depth is fixed
max_depth_learning_rate_fixed = np.array([(0, m) for m in sample_l1])       #Create a np.array where max_depth varies and learning_rate is fixed

eln = ElasticNet(alpha=0, l1_ratio=0 )

