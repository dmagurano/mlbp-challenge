import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy_indexed as npi

def visualize_errors_by_genre(y_true, y_predict, y_label_names):
    # Create dictionary of label occurences
    genre, count = np.unique(y_true, return_counts=True)
    dict_label_count = dict(zip(genre, count))

    # Count correct predictions per class
    correct_predictions_per_class = dict(zip(np.arange(1, 11), [0 for x in range(1, 11)]))
    for prediction, label in zip(y_predict, y_true):
        if (prediction == label):
            correct_predictions_per_class[prediction] += 1

            # Calculate wrong predictions
    correct_predictions = np.array(list(correct_predictions_per_class.values()))
    wrong_predictions = np.subtract(np.array(list(dict_label_count.values())), correct_predictions)
    print("Correct predictions per genre: ", correct_predictions)
    print("Wrong predictions per genre: ", wrong_predictions)

    # plot barchart
    sns.set(color_codes=True)
    fig = plt.subplots()
    ax = plt.subplot(111)
    index = np.arange(len(correct_predictions_per_class))
    p1 = ax.bar(index, correct_predictions, color='#7fbf7f')
    p2 = ax.bar(index, wrong_predictions, bottom=correct_predictions, color='#ff7f7f')

    ax.set_title("Errors of predictions by genre")
    ax.set_xlabel("Music genre")
    ax.set_ylabel("Amount of errors")
    ax.set_xticks(index)
    ax.set_xticklabels(y_label_names)
    ax.legend((p1[0], p2[0]), ('Correct', 'Wrong'))

    plt.show()


def confusion_matrix(y_true, y_pred, labels):
    cm = metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels,
                         columns=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def precision(pred, labels):
    prec = metrics.precision_score(pred, labels, average='macro')
    return prec


def recall(pred, labels):
    rec = metrics.recall_score(pred, labels, average='macro')
    return rec


def accuracy(pred, labels):
    acc = metrics.accuracy_score(pred, labels)
    return acc


def visualize_gridsearch(results, pipe_step, parameter):
    plt.figure(figsize=(13, 13))
    plt.title("Evaluation of the Parameter %s" % (parameter),
              fontsize=16)

    plt.xlabel(parameter)
    plt.ylabel("Average Score")
    plt.grid()

    ax = plt.axes()

    # Get the regular numpy array from the MaskedArray

    X_axis = np.array(results['param_%s' %(pipe_step)].data, dtype=float)

    for sample, style in (('train', '--'), ('test', '-')):
        x_unique, sample_score_mean = npi.group_by(X_axis).mean(results['mean_%s_score' % (sample)])
        x_unique, sample_score_std = npi.group_by(X_axis).mean(results['std_%s_score' % (sample)])
        ax.fill_between(x_unique, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0)
        ax.plot(x_unique, sample_score_mean, style,
                alpha=1 if sample == 'test' else 0.7,
                label=sample)

    best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
    best_score = results['mean_test_score'][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ]*2, [0, best_score],
            linestyle='-.', marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.show()