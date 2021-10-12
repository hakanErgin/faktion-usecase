import os
import pandas as pd
import numpy as np

from sklearn import metrics
from utils.visualize_substractions import visualize_substractions


def classifier_a_step2(path):
    df = pd.DataFrame(columns=["id", "result"])
    ids = []
    datas = []
    for filename in os.listdir(path):
        id = filename[:-4]
        res = visualize_substractions(inputfile=path+filename)
        ids.append(id)
        datas.append(res)
        dfrow = [id]
        dfrow.append(res)
        df.loc[len(df)] = dfrow

    return df


def report():
    df_normal = classifier_a_step2(path='./assets/normal_dice/0/')
    for i in range(1, 11):
        df_normal = pd.DataFrame.append(df_normal, classifier_a_step2(path='./assets/normal_dice/'+str(i)+'/'))

    df_normal['y_true'] = 0
    df_normal['y_pred'] = np.where(df_normal['result'] < 0.04, 0, 1)

    df_anomalous = classifier_a_step2(path='./assets/anomalous_dice/')
    df_anomalous['y_true'] = 1
    df_anomalous['y_pred'] = np.where(df_anomalous['result'] < 0.04, 0, 1)

    df_global = pd.concat([df_normal, df_anomalous])

    report = metrics.classification_report(df_global['y_true'], df_global['y_pred'])
    confus_matrix = metrics.confusion_matrix(df_global['y_true'], df_global['y_pred'])
    print(report)
    print(confus_matrix)
    # return report, confus_matrix
