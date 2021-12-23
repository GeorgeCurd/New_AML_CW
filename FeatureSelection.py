from pandas import read_csv, Series
import numpy as np
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from DataProcessing import test_Y, train_Y, test_normX, train_normX, train_X, test_X
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import Normalizer
from mlxtend.feature_selection import SequentialFeatureSelector as sfs



# load the encoder model from file
encoder = load_model('encoder.h5')
# encode the train data and re-normalise
X_train_encode = encoder.predict(train_normX)
# scaler = Normalizer().fit(X_train_encode)
# X_train_encode = scaler.transform(X_train_encode)
# encode the test data and re-normalise
X_test_encode = encoder.predict(test_normX)
# scaler = Normalizer().fit(X_test_encode)
# X_test_encode = scaler.transform(X_test_encode)

# Merge the extracted (encoded and normalised) features with original (normalised) features
X_train_merge = np.column_stack([X_train_encode, train_normX])
# encode the test data, and merge with feature selected data
X_test_merge = np.column_stack([X_test_encode, test_normX])
print(X_train_merge.shape)
print(X_test_merge.shape)

#Identify key features using random forest/extra trees
model = ExtraTreesClassifier()
model.fit(X_train_merge, train_Y)
reduced = SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=20)
train_normX_ETC_FS = reduced.transform(X_train_merge)
test_normX_ETC_FS = reduced.transform(X_test_merge)
print(train_normX_ETC_FS.shape)
print(test_normX_ETC_FS.shape)

# Using ANOVA F-Class
model = f_classif(X_train_merge, train_Y)
reduced = SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=20)
train_normX_AVOVA_FS = reduced.transform(X_train_merge)
test_normX_AVOVA_FS = reduced.transform(X_test_merge)
print(train_normX_AVOVA_FS.shape)
print(test_normX_AVOVA_FS.shape)

# Using forward selection
sfs1 = sfs(clf, k_features=5, forward=True, floating=False, verbose=2, scoring='accuracy',cv=5)
sfs1 = sfs1.fit(X_train, y_train)


