import pandas as pd
import numpy as np                              
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


(X_train,y_train),(X_test,y_test)=mnist.load_data()

print(X_train.shape)

le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)



X_train = X_train.reshape((X_train.shape[0], 28*28))
X_test = X_test.reshape((X_test.shape[0], 28*28))

classifier=RandomForestClassifier()

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

X=np.concatenate((X_train,X_test), axis=0)
y=np.concatenate((y_train,y_test), axis=0)

from sklearn.model_selection import cross_val_score
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
score = cross_val_score(rf, X, y)
print (np.mean(score))

