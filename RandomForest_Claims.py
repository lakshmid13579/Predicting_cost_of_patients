# Random Fprests

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

X = dataset.iloc[:, 0:13].values                                 
y = dataset.iloc[:, 13].values
                
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.preprocessing import scale
y = scale(y, axis=0, with_mean=True, with_std=True, copy=True) 
                
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = classifier_RF.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm_RF = confusion_matrix(y_test, y_pred_RF)
ac_RF = accuracy_score(y_test, y_pred_RF)