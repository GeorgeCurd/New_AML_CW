from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from FeatureSelection import train_normX_ETC_FS, test_normX_ETC_FS
from DataProcessing import train_Y, test_Y, train_normX, test_normX


# Run baseline log reg model
model = LogisticRegression(max_iter=10000)
# fit model on training set
model.fit(train_normX, train_Y)
# make prediction on test set
yhat = model.predict(test_normX)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
print("base LR model acc:", acc)

# Run log reg model with weighted feature selection
# define the model
model = LogisticRegression(max_iter=10000,penalty='none', solver='sag')
# fit the model on the training set
model.fit(train_normX_ETC_FS, train_Y)
# make predictions on the test set
yhat = model.predict(test_normX_ETC_FS)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
conf = confusion_matrix(test_Y, yhat)
print('WFS LR model acc::',acc)
print('WFS LR model CM:', conf)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('WFS LR model TPR/DR:', tp/(tp+fn))
print('WFS LR model FPR/FA:', fp/(tn+fp))







