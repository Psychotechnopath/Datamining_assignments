from standard_svm import surrogate_model_fitter, calculate_rf_input, load_nav_data
import pickle
import numpy as np

c_points = np.logspace(-12,12, 25)                                      #Create a range of parameter c, on a logarithmic scale, from 1e-12 to 1e12
gamma_points = np.logspace(-12, 12, 25)                                 #Create a range of parameter gamma, on a logarithmic scale, from 1e-12 to 1e12


with open('final_param_list_svm.pkl', 'rb') as f:
    final_param_list_loaded = pickle.load(f)


scores = surrogate_model_fitter(np.array(final_param_list_loaded), 'accuracy')  #Calculate scores of SVC from the final parameter list of previous question
svc_scores = calculate_rf_input(scores, classification=True)                    #Calculate the means of the three kfolds

with open('svc_scores_10.pkl', 'wb') as f:                            #Save these results to disk
    pickle.dump(svc_scores, f)

with open ('svc_scores_10.pkl', 'rb') as f:
    svc_scores_loaded = pickle.load(f)

print(final_param_list_loaded)
print(svc_scores_loaded)

max_ten_scores = np.argsort(svc_scores_loaded)[-10:][::-1]          #Get indices at which maximum score values reside
max_parameters_svc = final_param_list_loaded[max_ten_scores]        #Use these indices to return the 10 parameter combination values that have the highest scores

import pandas as pd
X_nav_pandas = pd.DataFrame(X_nav).drop([3,6,9,12,15,18] ,axis=1) #Convert to a dataframe, to easily remove 6 colums (6 sensor data).
print(X_nav_pandas.shape)

#We then use this list instead of the random_Hyperparameter list in Q2
def q2_callstack():
    pass
