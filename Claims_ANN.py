#Import libraries:
import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics   #Additional scklearn functions

# Importing the dataset
dataset = pd.read_csv('ClaimsData.csv')

#Finding the distribution of the variables
dataset[dataset.dtypes[(dataset.dtypes=="float64")|(dataset.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])
                
#Finding if there is any correlation between the variables               
corr= dataset.corr(method='spearman')

#Data Preprocessing
dataset = dataset.drop(['reimbursement2008', 'reimbursement2009'], axis=1)

X = dataset.iloc[:, 0:13]                            
y = dataset.iloc[:, 13]
y.describe()

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
                
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
                        
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

y_train = keras.utils.to_categorical(y_train)
y_train= y_train[:, 1:]
y_test = keras.utils.to_categorical(y_test)
y_test= y_test[:, 1:]

#X_train = X_train[:,:].values
#X_test = X_test.values

# Initialising the ANN
classifier_ann = Sequential()

# Adding the input layer and the first hidden layer
classifier_ann.add(Dense(13, input_dim=13, activation ='relu', kernel_initializer = 'uniform'))

# Adding the second hidden layer
classifier_ann.add(Dense(13, activation = 'relu',kernel_initializer = 'uniform'))

# Adding the output layer
classifier_ann.add(Dense(5, activation = 'softmax',kernel_initializer = 'uniform'))

# Compiling the ANN
classifier_ann.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier_ann.summary()
# Fitting the ANN to the Training set
classifier_ann.fit(X_train, y_train, batch_size= 10000, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred_ann = classifier_ann.predict(X_test)
b = np.zeros_like(y_pred_ann)
b[np.arange(len(y_pred_ann)), y_pred_ann.argmax(1)] = 1


# Making the Confusion Matrix
from keras.metrics import categorical_accuracy
k = categorical_accuracy(y_test, b)
b = b.astype(int)
b= pd.DataFrame(b)
len(b)
k= []
for i in range(0, len(b)):
    for j in range(0,5):
        if b[i,j]==1:
            k.append(j+1)
            break
        else:
            continue
k= pd.DataFrame(k)    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm_ann = confusion_matrix(y_test, k)
ac_ann = accuracy_score(y_test, k)

