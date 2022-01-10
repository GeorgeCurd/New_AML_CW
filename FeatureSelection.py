import pandas as pd
from pandas import read_csv, Series
import numpy as np
from sklearn.feature_selection import SelectFromModel, f_classif, RFE, SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from DataProcessing import test_Y, train_Y, test_normX, train_normX, train_X, test_X
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import time

start = time.time()

# load the encoder model from file
encoder = load_model('encoder.h5')
# encode the train data and re-normalise
X_train_encode = encoder.predict(train_normX)
# encode the test data and re-normalise
X_test_encode = encoder.predict(test_normX)

# Merge the extracted (encoded and normalised) features with original (normalised) features
X_train_merge = np.column_stack([X_train_encode, train_normX])
# encode the test data, and merge with feature selected data
X_test_merge = np.column_stack([X_test_encode, test_normX])
print(X_train_merge.shape)
print(X_test_merge.shape)
features_etc = 10
features_an = 10

#Identify key features using random forest/extra trees
model = ExtraTreesClassifier(random_state=1)
model.fit(X_train_merge, train_Y)
reduced = SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=features_etc)
train_normX_ETC_FS = reduced.transform(X_train_merge)
test_normX_ETC_FS = reduced.transform(X_test_merge)
print(train_normX_ETC_FS.shape)
print(test_normX_ETC_FS.shape)
print(np.array(train_normX_ETC_FS).var())
# Find out which features selected
feature_idx = reduced.get_support()
df = pd.DataFrame(X_train_merge)
feature_name = df.columns[feature_idx]
print(feature_name)

# Using ANOVA F-Class
# configure to select all features
fs = SelectKBest(score_func=f_classif, k=features_an)
# learn relationship from training data
fs.fit(X_train_merge, train_Y)
# transform train input data
train_normX_ANOVA_FS = fs.transform(X_train_merge)
# transform test input data
test_normX_ANOVA_FS = fs.transform(X_test_merge)
print(train_normX_ANOVA_FS.shape)
print(test_normX_ANOVA_FS.shape)

# Find out which features selected
print(fs.get_support([1]))


# Using RFE
# create pipeline
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=features)
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# fit the model on all available data
pipeline.fit(X_train_merge, train_Y)
# transform train input data
train_normX_RFE_FS = rfe.transform(X_train_merge)
# transform test input data
test_normX_RFE_FS = rfe.transform(X_test_merge)
print(train_normX_RFE_FS.shape)
print(test_normX_RFE_FS.shape)
