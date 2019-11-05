from pbreg import ProbabilisticRandomForestRegressor
from pbreg import EI
import numpy as np
from matplotlib import pyplot as plt
import openml as oml
from sklearn.svm import SVC
from sklearn.model_selection import  cross_validate
import pickle
np.random.seed(47)

def load_nav_data():
    robotnav = oml.datasets.get_dataset(1497)
    X_nav, y_nav, cats, attrs = robotnav.get_data(dataset_format='array',
    target=robotnav.default_target_attribute)
    return X_nav, y_nav

def load_reg_data():
    robotarm = oml.datasets.get_dataset(189)
    Xr, yr, catsr, attrsr = robotarm.get_data(dataset_format='dataframe',
    target=robotarm.default_target_attribute)
    return Xr, yr


X_nav, y_nav = load_nav_data()
Xr, yr = load_reg_data()
c_points = np.logspace(-12,12, 25)                                      #Create a range of parameter c, on a logarithmic scale, from 1e-12 to 1e12
gamma_points = np.logspace(-12, 12, 25)                                 #Create a range of parameter gamma, on a logarithmic scale, from 1e-12 to 1e12
sample_c = np.random.choice(c_points, 10 , replace=False)               #Sample 10 random points out of the parameter c range
sample_g = np.random.choice(gamma_points, 10, replace=False)            #Sample 10 random points out of the parameter gamma range

random_hyperparams = np.array(list(zip(sample_c, sample_g)))            #Zip together sample_c and sample_g, cast it to the right data structure (np.array)
c_points_gamma_fixed = np.array([(c, 1/24) for c in c_points])          #Create a np.array where c varies and gamma is fixed
gamma_points_c_fixed = np.array([(1, gamma) for gamma in gamma_points]) #Create a np.array where gamma varies and c is fixed


def surrogate_model_fitter(random_params, scoring_metric, model_name):
    score_list = []
    for param1, param2 in random_params:
        svc = SVC(C=param1, kernel='rbf', gamma= param2)
        score = cross_validate(svc, X_nav, y_nav, cv=3, scoring=[scoring_metric])
        score_list.append(score)
    with open("score_list_{}.pkl".format(model_name), "wb") as f:
        pickle.dump(score_list, f)


def load_surrogate_score(model_name):
    with open('score_list_{}.pkl'.format(model_name), 'rb') as score_list:
        score_list_model = pickle.load(score_list)
    return score_list_model


def calculate_rf_input(score_list_model, classification=True):
    if classification:
        rf_input = np.array([1 - np.mean(i['test_accuracy']) for i in score_list_model]) #Calculate the means out of the returned accuracy from the standard svm
    else:
        rf_input = np.array([abs(np.mean(i['neg_mean_squared_error'])) for i in score_list_model])
    return rf_input


def random_forest_fitter(score_to_fit, random_params, fixed_slice):
    rf = ProbabilisticRandomForestRegressor(n_estimators=100)
    rf.fit(random_params, score_to_fit)
    Y_pred, sigma = rf.predict(fixed_slice, return_std=True)
    return Y_pred, sigma, rf


def surr_acq_plotter(param_x_axis, fixed_slice, y_pred_param, sigma_param, rf_param, x_axis_text):
    # Plot surrogate model
    ei_score = EI(rf_param, fixed_slice)
    plt.subplot(2,1,1)
    plt.ylabel('Surrogate model error')
    plt.xscale('log')
    plt.plot(param_x_axis, y_pred_param, 'r.' , markersize=10, label=u'Observations')
    plt.plot(param_x_axis, y_pred_param, label=u'Prediction')
    plt.fill_between(param_x_axis.ravel(),y_pred_param-2*sigma_param,y_pred_param+2*sigma_param,alpha=0.1,label='Uncertainty')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplot(2,1,2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('{}'.format(x_axis_text))
    plt.ylabel('Expected Improvement')
    plt.plot(param_x_axis, ei_score)
    plt.show()
    return ei_score

def q1_callstack(model_name: str, random_param: np.array, scoring_metric: str,  fixed_slice: np.array, param_x_axis, x_axis_text: str, classification = True):
    surrogate_model_fitter(random_hyperparams, scoring_metric, model_name)  #Fit the surrogate model on random hyperparams
    score = load_surrogate_score(model_name)                                #Load the surrogate model from disk (Functionality was built if models surrogate models take long time to fit)
    rf_score = calculate_rf_input(score, classification)                    #Calculate the randomforest inputs, If classification is True calculate accuracy means, otherwise positive MSE
    y, sig, rf = random_forest_fitter(rf_score, random_param, fixed_slice)  #Fit the random forest on the random parameters, make it predict for the slice we want
    ei_score = surr_acq_plotter(param_x_axis, fixed_slice, y, sig, rf, x_axis_text)
    return ei_score


#Save Expected improvement scores for use in additional iterations
ei_score_cvaried_gamma_fixed  = q1_callstack('svm', random_hyperparams, 'accuracy', c_points_gamma_fixed, c_points, 'C varied, Gamma fixed at Gamma=1/24', classification=True)
ei_score_gamma_varied_c_fixed = q1_callstack('svm', random_hyperparams, 'accuracy', gamma_points_c_fixed, gamma_points, 'Gamma varied, C fixed at C=1 ',classification=True)


# with open("ei_score_cvaried_gamma_fixed.pkl", "wb") as f:
#     pickle.dump(ei_score_cvaried_gamma_fixed, f)
#
# with open("ei_score_gamma_varied_c_fixed", "wb") as f:
#     pickle.dump(ei_score_gamma_varied_c_fixed, f)
#
#
# with open("ei_score_cvaried_gamma_fixed.pkl", "rb") as f:
#     ei1 = pickle.load(ei_score_cvaried_gamma_fixed, f)
#
# with open("ei_score_gamma_varied_c_fixed", "rb") as f:
#     ei2 = pickle.load(ei_score_gamma_varied_c_fixed, f)

# max_ei_c = np.unravel_index(np.argmax(ei_score_cvaried_gamma_fixed, axis=None), shape=ei_score_cvaried_gamma_fixed.shape)
# best_c = c_points[max_ei_c]
# max_ei_gamma = np.unravel_index(np.argmax(ei_score_gamma_varied_c_fixed, axis=None), shape=ei_score_gamma_varied_c_fixed.shape)
# best_gamma = gamma_points[max_ei_gamma]



def q1_3it_callstack(model_name: str,
                     random_param: np.array,
                     scoring_metric: str,
                     fixed_slice_1: np.array, fixed_slice_2: np.array,
                     param_x_axis_1: np.array, param_x_axis_2: np.array,
                     x_axis_text1: str, x_axis_text2: str,
                     classification = True):
    """First, we calculate the optimal hyperparameter setup from the EI results in the previous question. Then we repeat the procedure 3 times. This gives us a total of 4 iterations"""
    max_ei_param_1 = np.unravel_index(np.argmax(ei_score_cvaried_gamma_fixed, axis=None), shape=ei_score_cvaried_gamma_fixed.shape)
    best_param1_value_1 = param_x_axis_1[max_ei_param_1]
    max_ei_param_2 = np.unravel_index(np.argmax(ei_score_gamma_varied_c_fixed,  axis=None), shape=ei_score_gamma_varied_c_fixed.shape)
    best_param1_value_2 = param_x_axis_2[max_ei_param_2]

    new_param_list = np.vstack((random_param, np.array(best_param1_value_1, best_param1_value_2)))
    for i in range(3):
        surrogate_model_fitter(new_param_list, scoring_metric, model_name)
        score = load_surrogate_score(model_name)
        rf_score = calculate_rf_input(score, classification)

        y1, sig1, rf1 = random_forest_fitter(rf_score, new_param_list, fixed_slice_1)
        ei_score1 = surr_acq_plotter(param_x_axis_1, fixed_slice_1, y1, sig1, rf1, x_axis_text1)
        max_ei_param_1 = np.unravel_index(np.argmax(ei_score1, axis=None), shape=ei_score1.shape)
        best_param1_value_1 = c_points[max_ei_param_1]

        y2, sig2, rf2 = random_forest_fitter(rf_score, new_param_list, fixed_slice_2)
        ei_score2 = surr_acq_plotter(param_x_axis_2, fixed_slice_2, y2, sig2, rf2, x_axis_text2)
        max_ei_param_2 = np.unravel_index(np.argmax(ei_score2, axis=None), shape=ei_score2.shape)
        best_param1_value_2 = gamma_points[max_ei_param_2]
        new_param_list = np.vstack((new_param_list, np.array(best_param1_value_1, best_param1_value_2)))

q1_3it_callstack('svm', random_hyperparams, 'accuracy', c_points_gamma_fixed, c_points, gamma_points_c_fixed, gamma_points,
                 "C varied, gamma fixed at Gamma =1/24", "Gamma varied, C fixed at C=1 ", classification=True)

# # def q1_3it_callstack():
# #
# # def q1_30_it_callstack():
# #     pass
# #     # test = np.vstack((random_hyperparams, np.array([1,1])))
# #     # print(test)
# #     #
# #     # ind = np.unravel_index(np.argmax(EI(rf, gamma_points_c_fixed), axis=None), shape=EI(rf, gamma_points_c_fixed).shape)
# #     # print(c_points[ind])
