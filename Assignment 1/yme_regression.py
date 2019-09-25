from sklearn.model_selection import train_test_split, cross_validate
import openml as oml
from sklearn.linear_model import LogisticRegression
import time
from matplotlib import pyplot as plt


SVHN = oml.datasets.get_dataset(41081)                                   #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)


X_90_percent, X_10_percent, y_90_percent, y_10_percent = train_test_split(X, y, test_size=0.1, stratify=y, random_state=47)

logisticReg = LogisticRegression()

test_accuracy_list = []
test_accuracy_std_list = []
time_execution_list = []
training_percentages = [1,2,3,4,5,6,7,8,9,10]

for i in range(1,11):
    start = time.time()
    scores = cross_validate(logisticReg, X_10_percent[:i*992], y_10_percent[0:i*992], cv=3, scoring=['accuracy'], n_jobs=-1)
    test_accuracy_list.append(scores['test_accuracy'])
    test_accuracy_std_list.append(scores['test_accuracy'].std())
    stop = time.time()
    duration = start-stop
    time_execution_list.append(duration)
    print("Training complete on {}% subsample of data".format(i))


plt.subplot(2,1,1)
plt.plot(training_percentages, test_accuracy_list, '-o')
plt.title('Testing accuracy and training times plot')
plt.xlabel("Percentage of data used")
plt.ylabel("Testing accuracy")

plt.subplot(2,1,2)
plt.plot(training_percentages,time_execution_list , '-o')
plt.xlabel("Percentage of data used")
plt.ylabel("Execution time in seconds")
plt.show()


