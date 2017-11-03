# Ordered Logistic Regression

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
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.preprocessing import scale
y_train = scale( y_train, axis=0, with_mean=True, with_std=True, copy=True )               
            
#Fitting OrderedLogistic Regression
from scipy import stats
import mord as m
c = m.OrdinalRidge()
c.fit(X_train, y_train)
y_pred = c.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
                 
                 

