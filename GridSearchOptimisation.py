from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from FeatureSelection import train_normX_ETC_FS, test_normX_ETC_FS
from DataProcessing import train_Y, test_Y, train_normX, test_normX
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


# Grid search on log regression model
define models and parameters
model = LogisticRegression(max_iter=10000)
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalty = ['none', 'l1', 'l2', 'elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train_encode, train_Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# # Run Grid Search on SVC
# model = SVC()
# # grid = {'C': [0.1, 1, 10, 100, 1000],
# #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
# #               'kernel': ['rbf']}
#
# grid = {'C': [100],
#               'gamma': [0.1],
#               'kernel': ['rbf']}
#
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(train_normX_ETC_FS, train_Y)
# # Summarise
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
