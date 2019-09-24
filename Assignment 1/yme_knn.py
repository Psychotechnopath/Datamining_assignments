import time
from sklearn.model_selection import cross_validate, train_test_split
import openml as oml
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

SVHN = oml.datasets.get_dataset(41081)                                                 #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_90_percent, X_10_percent, y_90_percent, y_10_percent = train_test_split(X, y, test_size=0.1, stratify=y, random_state=47)

knn = KNeighborsClassifier()
test_accuracy_list = []
test_accuracy_std_list = []
time_execution_list = []
training_percentages = [1,2,3,4,5,6,7,8,9,10]

for i in range(1,11):
    start = time.time()
    scores = cross_validate(knn, X_10_percent[:i*992], y_10_percent[0:i*992], cv=3, scoring=['accuracy'])
    test_accuracy_list.append(scores['test_accuracy'])
    test_accuracy_std_list.append(scores['test_accuracy'].std())
    stop = time.time()
    duration = start-stop
    time_execution_list.append(duration)
    print("Training complete on {}% subsample of data".format(i))


plt.subplot(2,1,1)
plt.plot(training_percentages, test_accuracy_list)
plt.title('Testing accuracy and training times plot', '-o')
plt.xlabel("Percentage of data used")
plt.ylabel("Testing accuracy")

plt.subplot(2,1,2)
plt.plot(training_percentages,time_execution_list , '-o')
plt.xlabel("Percentage of data used")
plt.ylabel("Execution time in seconds")
plt.show()


