# Step 1 : Import relevent packages 

from sklearn import svm

# Packages for Analysis

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Packages for Visuals
#%matplotlib inline

# Step 2: Import Dataset
streetview_data = pd.read_csv('svhn.csv')

# Step 3: Separate the data with features and label
X = streetview_data.iloc[:,:-1].values # selecting all the records from all the fields except last field
y = streetview_data.iloc[:,3072].values # Selecting all the records from last field

#Generate 10% Sub-sample of data
X_sub = X[0:9920]                                                        
y_sub = y[0:9920]

# step 4: Train/Test split
# Training Data --> 75%
# Test Data --> 25%

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_sub, y_sub,
                                                 test_size = 0.25,
                                                 random_state = 0)

# Step 5: Feature Scaling ----- Applied over the feature data
# Generalize the magnitude

from sklearn.preprocessing import StandardScaler # Normal distribution
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# Step 6: Training model using SVM
from sklearn.model_selection import GridSearchCV

hyper_parameters = {'kernel':('sigmoid','rbf'), 'C':[0.1,1]}
svc = svm.SVC(gamma="scale")

clf = GridSearchCV(estimator=svc,
                   param_grid=hyper_parameters,
                   cv=5)

clf.fit(X_train, y_train)

# Step 7:# Making Prediction

predicted = clf.predict(X_test)

# Step 8:# Check for Accuracy

score = clf.score(X_test, y_test)

# step 9: Make Confusion matrix using seaborn
from sklearn import metrics

cm = metrics.confusion_matrix(y_test,predicted)

plt.figure(figsize = (7,7))
sns.heatmap(cm, annot=True,square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
