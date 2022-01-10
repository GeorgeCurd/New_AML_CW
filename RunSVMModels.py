from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from FeatureSelection import start, train_normX_ETC_FS, test_normX_ETC_FS  # train_normX_ANOVA_FS, test_normX_ANOVA_FS ,
from DataProcessing import train_Y, test_Y, train_normX, test_normX
import time

# # Run baseline SVC model
model = SVC(kernel='rbf')
# fit model on training set
model.fit(train_normX_ETC_FS, train_Y)
# make prediction on test set
yhat = model.predict(test_normX_ETC_FS)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
print("base SVM model acc:", acc)
conf = confusion_matrix(test_Y, yhat)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('base SVM model TPR/DR:', tp/(tp+fn))
print('base SVM model FPR/FA:', fp/(tn+fp))

# # Run SVC model - with ANOVA
#, C=100, gamma=0.001
model = SVC(kernel='rbf', C=50, gamma=0.002) #0.0.955276657204044, c=50, gamma= 0.002, feat=20
# fit model on training set
model.fit(train_normX_ANOVA_FS, train_Y)
# make prediction on test set
yhat = model.predict(test_normX_ANOVA_FS)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
mcc = matthews_corrcoef(test_Y, yhat)
print("ANOVA SVM model acc:", acc)
conf = confusion_matrix(test_Y, yhat)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('ANOVA SVM model TPR/DR:', tp/(tp+fn))
print('ANOVA SVM model FPR/FA:', fp/(tn+fp))
print('ANOVA SVM model FP/Type1:', fp)
print('ANOVA SVM model FN/Type2:', fn)
print('ANOVA SVM model MCC:', mcc)

# Calculate TTR
end = time.time()
runtime = end - start
print('ANOVA SVM model TTR:', runtime)


#Run SVC model - with ETC WFS
#
model = SVC(kernel='rbf', C=1, gamma=0.002)
# fit model on training set
model.fit(train_normX_ETC_FS, train_Y)
# make prediction on test set
yhat = model.predict(test_normX_ETC_FS)
# calculate accuracy
acc = accuracy_score(test_Y, yhat)
mcc = matthews_corrcoef(test_Y, yhat)
print("ETC SVM model acc:", acc)
conf = confusion_matrix(test_Y, yhat)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('ETC SVM model TPR/DR:', tp/(tp+fn))
print('ETC SVM model FPR/FA:', fp/(tn+fp))
print('ETC SVM model FP/Type1:', fp)
print('ETC SVM model FN/Type2:', fn)
print('ETC SVM model MCC:', mcc)

# Calculate TTR
end = time.time()
runtime = end - start
print('ETC SVM model TTR:', runtime)
