from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from FeatureSelection import train_normX_ETC_FS, test_normX_ETC_FS
from DataProcessing import train_Y, test_Y, train_normX, test_normX

# # Run baseline SVC model
# model = SVC()
# # fit model on training set
# model.fit(train_normX, train_Y)
# # make prediction on test set
# yhat = model.predict(test_normX)
# # calculate accuracy
# acc = accuracy_score(test_Y, yhat)
# print("base SVC model acc:", acc)


# Run SVC model with weighted feature selection
# define the model
model = SVC(C=100, gamma=0.1)
# fit the model on the training set
model.fit(train_normX_ETC_FS, train_Y)
# make predictions on the test set
yhat = model.predict(test_normX_ETC_FS)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
conf = confusion_matrix(test_Y, yhat)
print('WFS SVC model acc::',acc)

