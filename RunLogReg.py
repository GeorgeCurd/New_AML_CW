from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from FeatureSelection import start, test_normX_ANOVA_FS, train_normX_ANOVA_FS # train_normX_ETC_FS, test_normX_ETC_FS ,
from DataProcessing import train_Y, test_Y, train_normX, test_normX
import time

# Run baseline log reg model
model = LogisticRegression(max_iter=10000)
# fit model on training set
model.fit(train_normX, train_Y)
# make prediction on test set
yhat = model.predict(test_normX)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
print("base LR model acc:", acc)
conf = confusion_matrix(test_Y, yhat)
print('base LR model acc::',acc)
print('base LR model CM:', conf)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('base LR model TPR/DR:', tp/(tp+fn))
print('base LR model FPR/FA:', fp/(tn+fp))

# Run log reg model with ANOVA weighted feature selection
# define the model
#, solver='newton-cg'
model = LogisticRegression(max_iter=10000, penalty='l2', C=1000)
# fit the model on the training set
model.fit(train_normX_ANOVA_FS, train_Y)
# make predictions on the test set
yhat = model.predict(test_normX_ANOVA_FS)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
mcc = matthews_corrcoef(test_Y, yhat)
conf = confusion_matrix(test_Y, yhat)
print('ANOVA LR model acc::',acc)
print('ANOVA LR model CM:', conf)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('ANOVA LR model TPR/DR:', tp/(tp+fn))
print('ANOVA LR model FPR/FA:', fp/(tn+fp))
print('ANOVA LR model FP/Type1:', fp)
print('ANOVA LR model FN/Type2:', fn)
print('ANOVA LR model MCC:', mcc)
#
# Calculate TTR
end = time.time()
runtime = end - start
print('ANOVA LR model TTR:', runtime)

#Run log reg model with ETC tuned weighted feature selection
#define the model

model = LogisticRegression(max_iter=10000, penalty='l2', C=400) #0.9682504108770357
# fit the model on the training set
model.fit(train_normX_ETC_FS, train_Y)
# make predictions on the test set
yhat = model.predict(test_normX_ETC_FS)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
mcc = matthews_corrcoef(test_Y, yhat)
conf = confusion_matrix(test_Y, yhat)
print('ETC LR model acc:',acc)
print('ETC LR model CM:', conf)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('ETC LR model TPR/DR:', tp/(tp+fn))
print('ETC LR model FPR/FA:', fp/(tn+fp))
print('ETC LR model FP/Type1:', fp)
print('ETC LR model FN/Type2:', fn)
print('ETC LR model MCC:', mcc)
#
# Calculate TTR
end = time.time()
runtime = end - start
print('ETC LR model TTR:', runtime)
