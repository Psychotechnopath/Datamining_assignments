import pickle
from matplotlib import pyplot as plt

results_file = open('knn_coarse.pkl','rb')
cv_results_ = pickle.load(results_file)
results_file.close()

param_grid = [5,15,25,35,45] #List used to set the tested Hyper parameter range on the x-axis


plt.subplot(2,1,1)
plt.title('Training and testing accuracies for KNN in a Grid Search')
plt.plot(param_grid, cv_results_['split0_train_accuracy'], '-o')
plt.plot(param_grid, cv_results_['split1_train_accuracy'], '-o')
plt.plot(param_grid, cv_results_['split2_train_accuracy'], '-o')
plt.legend(['Fold 1', 'Fold 2', 'Fold 3'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel("Number of neighbors")
plt.ylabel("Training Accuracy")

plt.subplot(2,1,2)
plt.plot(param_grid, cv_results_['split0_test_accuracy'], '-o')
plt.plot(param_grid, cv_results_['split1_test_accuracy'], '-o')
plt.plot(param_grid, cv_results_['split2_test_accuracy'], '-o')
plt.ylabel("Testing Accuracy")

plt.show()
