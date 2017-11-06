import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import evaluation

from sklearn import preprocessing
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

### Defining Pipeline###
model_logistic = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
pca = decomposition.PCA()

pipe = Pipeline(steps=[('pca', pca), ('logistic', model_logistic)])

n_components = [254]#[5, 10, 50, 100, 150, 200, 254]
Cs = np.logspace(-5, 1, 15)

# Execution of Cross validation
scoring = 'accuracy'
# scoring = 'neg_log_loss'
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs),
                         scoring=scoring,
                         cv=3,
                         verbose=10)

estimator.fit(X_train_normalized, y_train_row)

#print(estimator.cv_results_)
print(estimator.best_score_)
print(estimator.best_params_)

### Plot the best number of components ###
pca.fit(X_train_normalized)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance')
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()

### Evaluation ###
prediction = estimator.predict(X_train_normalized)
evaluation.visualize_errors_by_genre(y_train_row, prediction, y_label_names)
evaluation.confusion_matrix(y_train_row, prediction, y_label_names)

###Visualize grid search
results = estimator.cv_results_
evaluation.visualize_gridsearch(results,'logistic__C', 'C')
#evaluation.visualize_gridsearch(results,'pca__n_components', 'n_components')