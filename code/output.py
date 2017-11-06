import pandas as pd
import numpy as np


def accuracy(prediction_test):
    prediction_test = pd.DataFrame(
        {"Sample_id": np.arange(1, (len(prediction_test) + 1)), "Sample_label": prediction_test})
    prediction_test.to_csv("prediction_of_X_test_accuracy.csv", index=False)


def logloss(prediction_test):
    col = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9',
           'Class_10']
    prediction_test = pd.DataFrame(prediction_test, columns=col)
    pred_id = pd.DataFrame({"Sample_id": np.arange(1, (len(prediction_test) + 1))})
    prediction_test = pd.concat([pred_id, prediction_test], axis=1)
    prediction_test.to_csv("prediction_of_X_test_logloss.csv", index=False)

