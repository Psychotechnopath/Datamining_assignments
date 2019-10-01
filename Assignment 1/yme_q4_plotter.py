import pickle
from matplotlib import pyplot as plt

results_file = open('C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 1/saved_files/logreg_fine.pkl','rb')
cv_results_ = pickle.load(results_file)
results_file.close()

param_grid_n = [5,15,25,35,45] #List used to set the tested Hyper parameter range on the x-axis
param_grid_c = [1e-5 ,1e-4, 1e-3, 1e-2 ,1e-1 ,1, 1e2, 1e3, 1e4, 1e5]
param_grid_c2 = [1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9, 1e12]

plt.subplot(2,1,1)
plt.title('Training and testing accuracies for Logistic Regression in a Grid Search')
plt.plot(param_grid_c, cv_results_['split0_train_accuracy'], '-o')
plt.plot(param_grid_c, cv_results_['split1_train_accuracy'], '-o')
plt.plot(param_grid_c, cv_results_['split2_train_accuracy'], '-o')
plt.legend(['Fold 1', 'Fold 2', 'Fold 3'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xscale('log')
plt.xlabel("C")
plt.ylabel("Training Accuracy")

plt.subplot(2,1,2)
plt.plot(param_grid_c, cv_results_['split0_test_accuracy'], '-o')
plt.plot(param_grid_c, cv_results_['split1_test_accuracy'], '-o')
plt.plot(param_grid_c, cv_results_['split2_test_accuracy'], '-o')
plt.xscale('log')
plt.ylabel("Testing Accuracy")

plt.show()
