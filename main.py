import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition, linear_model
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import evaluation
import output


def main():
    model="pca_svc"

    ## Read data sets
    header = pd.read_csv('Datasets/header.txt', delimiter=",", header=None)
    X_train = pd.read_csv('Datasets/train_data.csv', delimiter=",", header=None)
    X_train.columns = list(header.values)
    X_test = pd.read_csv('Datasets/test_data.csv', delimiter=",", header=None)
    X_test.columns = list(header.values)
    y_train = pd.read_csv('Datasets/train_labels.csv', delimiter=",", header=None)
    y_label_names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB', 'International', 'Country', 'Reggae', 'Blues']

    labels = y_train.values
    c, r = labels.shape
    labels = labels.reshape(c, )

    ## Feature normalization
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X_train, labels)
    scores.mean()


    clf = None

    if(model=="pca_svc"):
        ## Defining pipeline with PCA and SVC rbf hyperparameter tuning

        svm = SVC(kernel="rbf")
        pca = decomposition.PCA()
        pipe = Pipeline(steps=[('pca', pca), ('svc', svm)])

        n_components = [30]
        C_range = np.logspace(2, 6, 5)
        gamma_range = np.logspace(-9, -1, 5)

        svc_pipe = GridSearchCV(pipe,
                                 dict(pca__n_components=n_components,
                                      svc__C=C_range,
                                      svc__gamma=gamma_range),
                                 verbose=10,
                                 n_jobs=-1)
        svc_pipe.fit(X_train, labels)
        print(svc_pipe.best_score_)
        print(svc_pipe.best_params_)

        clf = svc_pipe.best_estimator_

    if(model == "rfe_log"):
        log = linear_model.LogisticRegressionCV(multi_class='multinomial', class_weight=None, cv=3)
        rfecv = RFECV(estimator=log, step=5, cv=2,
                      scoring='accuracy',
                      verbose=100,
                      n_jobs=-1)
        rfecv.fit(X_train, labels)

        print("Optimal number of features : %d" % rfecv.n_features_)

        X_train = rfecv.transform(X_train)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

        log_reg = linear_model.LogisticRegressionCV(multi_class='multinomial', class_weight=None, cv=3)
        log_reg.fit(X_train, labels)

        X_test = rfecv.transform(X_test)
        clf = log_reg

    ### Evaluation ###

    accuracy = clf.score(X_train, labels)
    print("Accuracy: {0}", format(accuracy))

    prediction_training = clf.predict(X_train)
    precision = evaluation.precision(prediction_training, labels)
    recall = evaluation.recall(prediction_training, labels)
    print("Precision: {0}", format(precision))
    print("Recall: {0}", format(recall))

    evaluation.confusion_matrix(labels, prediction_training, y_label_names)

    # ### Output ###

    prediction_test = clf.predict(X_test)
    output.accuracy(prediction_test)

if __name__ == '__main__':
    main()