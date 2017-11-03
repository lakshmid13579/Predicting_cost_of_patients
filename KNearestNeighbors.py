# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ClaimsData.csv')

#Finding the distribution of the variables
dataset[dataset.dtypes[(dataset.dtypes=="float64")|(dataset.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])
                
#Finding if there is any correlation between the variables               
corr= dataset.corr(method='spearman')

#Data Preprocessing
dataset = dataset.drop(['reimbursement2008', 'reimbursement2009'], axis=1)
dataset = dataset.sort_values(by='bucket2008', ascending=True)
X = dataset.iloc[:, 0:12].values                                 
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
