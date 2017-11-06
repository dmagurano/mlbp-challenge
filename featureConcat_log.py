import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import evaluation

from sklearn import preprocessing
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

## Read data sets
header = pd.read_csv('Datasets/header.txt', delimiter=",", header=None)
X_train = pd.read_csv('Datasets/train_data.csv', delimiter=",", header=None)
X_train.columns = list(header.values)
X_test = pd.read_csv('Datasets/test_data.csv', delimiter=",", header=None)
X_test.columns = list(header.values)
y_train = pd.read_csv('Datasets/train_labels.csv', delimiter=",", header=None)
y_train_row = np.ravel(y_train)
y_label_names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB', 'International', 'Country', 'Reggae', 'Blues']


### Preprocessing ###
X_train_normalized = preprocessing.scale(X_train)

# This dataset is way too high-dimensional. Better do PCA:
pca = decomposition.PCA(n_components=5)

# Maybe some original features where good, too?
selection = SelectKBest(k=5)

# Build estimator from PCA and Univariate selection:
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X_train_normalized, y_train_row).transform(X_train_normalized)


model_logistic = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
pca = decomposition.PCA()

pipe = Pipeline(steps=[("features", combined_features), ('logistic', model_logistic)])

n_components = [50, 150, 254]
k_select = [1, 3, 5]
Cs = np.logspace(-4, 0, 10)

# Execution of Cross validation
scoring = 'accuracy'
# scoring = 'neg_log_loss'
estimator = GridSearchCV(pipe,
                         dict(features__pca__n_components=n_components,
                              features__univ_select__k=k_select,
                              logistic__C=Cs),
                         scoring=scoring,
                         cv=3,
                         verbose=10)

estimator.fit(X_train_normalized, y_train_row)

print(estimator.cv_results_)
print(estimator.best_score_)

### Evaluation ###
prediction = estimator.predict(X_train_normalized)
evaluation.visualize_errors_by_genre(y_train_row, prediction, y_label_names)
evaluation.confusion_matrix(y_train_row, prediction, y_label_names)

# ###Visualize grid search
# results = estimator.cv_results_
# evaluation.visualize_gridsearch(results,'logistic__C', 'C')
# evaluation.visualize_gridsearch(results,'pca__n_components', 'n_components')