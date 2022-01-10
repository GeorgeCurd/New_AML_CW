from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from FeatureSelection import train_normX_ANOVA_FS, test_normX_ANOVA_FS, train_normX_ETC_FS, test_normX_ETC_FS
from DataProcessing import train_Y, test_Y, train_normX, test_normX

# Run baseline ETC model
model = ExtraTreesClassifier()
# fit the model on the training set
model.fit(train_normX_ETC_FS, train_Y)
# make predictions on the test set
yhat = model.predict(test_normX_ETC_FS)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
conf = confusion_matrix(test_Y, yhat)
print('base ETC model acc:',acc)
conf = confusion_matrix(test_Y, yhat)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('base ETC model TPR/DR:', tp/(tp+fn))
print('baseETC model FPR/FA:', fp/(tn+fp))

# Run ETC model with weighted feature selection
model = ExtraTreesClassifier(n_estimators=1000)
# fit the model on the training set
model.fit(train_normX_ANOVA_FS, train_Y)
# make predictions on the test set
yhat = model.predict(test_normX_ANOVA_FS)
# calculate classification accuracy
acc = accuracy_score(test_Y, yhat)
print('ANOVA ETC model acc::',acc)
conf = confusion_matrix(test_Y, yhat)
tp = conf[0][0]
fp = conf[0][1]
fn = conf[1][0]
tn = conf[1][1]
print('ANOVA ETC model TPR/DR:', tp/(tp+fn))
print('ANOVA ETC model FPR/FA:', fp/(tn+fp))
