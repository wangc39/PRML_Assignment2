import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, recall_score
from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix
from scipy import interp


def draw_ROC(y_test, y_proba=None, output_path=None):



    n_classes = len(list(set(y_test)))
    y_test = label_binarize(y_test, classes=list(set(y_test)))

    # print(y_test.shape)
    # print(y_proba.shape)


    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # print(i, roc_auc[i], y_test[:, i], y_proba[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # print(i, roc_auc[i], )
        # print(y_test[:, i]) # y_proba[:, i]
        # # print(y_proba[:, i]) # y_proba[:, i]
        # # print(y_test[:, i]) # y_proba[:, i]

        # print(np.argmax(y_test[:, i]))
        # print(np.argmax(y_proba[:, i]))

        # print('---'*20)

        # exit(0)
        # print("roc_auc[i]", roc_auc[i])


    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.6f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.6f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")

    plt.savefig(output_path)
    print('output_path')
    plt.show()


def model_evaluation(y_test, y_predict):
    print('recall: %.4f' % recall_score(y_test, y_predict, average='micro'))
    print('f1-score: %.4f' % f1_score(y_test, y_predict, average='micro'))


def show_classification_report(y_test, y_predict, target_names):
    print(classification_report(y_test, y_predict, target_names=target_names))

    pass


def show_confusion_matrix(y_test, y_predict, n_classes=40):
    print(confusion_matrix(y_test, y_predict, labels=range(n_classes)))