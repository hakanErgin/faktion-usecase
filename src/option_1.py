import os
import pandas as pd
import numpy as np
from sklearn import metrics
from utils.visualize_substractions import visualize_substractions


def dataframe_a_step2(path):
    """
    Builds a dataframe with a row for each picture in the input folder.
    The 3 columns are the id of the picture (its filename), result and result2 from utils.visualize_substractions
    :param path: path of the folder containing some pictures
    :return: a dataframe with 3 columns
    """
    df = pd.DataFrame(columns=["id", "result", "result2"])
    for filename in os.listdir(path):
        id = filename[:-4]
        res, res2 = visualize_substractions(inputfile=path+filename, plot=False)
        data = [id, res, res2]
        df.loc[len(df)] = data
    return df

def build():
    # building data frame for normal dice
    df_normal = dataframe_a_step2(path='./assets/normal_dice/0/')
    for i in range(1, 11):
        df_normal = pd.DataFrame.append(df_normal, dataframe_a_step2(path='./assets/normal_dice/'+str(i)+'/'))
    df_normal['y_true'] = 0

    # building data frame for anomalous dice
    df_anomalous = dataframe_a_step2(path='./assets/anomalous_dice/')
    df_anomalous['y_true'] = 1

    # concatenation of the 2 df
    df_global = pd.concat([df_normal, df_anomalous])

    # assigned class by the "model" (y_pred) based on result and result2 values that best discriminate the 2 classes
    df_global['y_pred'] = np.where((df_global['result'] < 0.045) & (df_global['result2'] < 0.14), 0, 1)

    # classification report and confusion matrix
    report = metrics.classification_report(df_global['y_true'], df_global['y_pred'])
    confus_matrix = metrics.confusion_matrix(df_global['y_true'], df_global['y_pred'])
    print(report, confus_matrix)
    return df_global


def classifier_a_step2(inputfile):
    """
    Uses function utils.visualize_substractions to classify between normal (0) and anomalous (1) a single picture.
    :param inputfile: the complete path to the picture.
    :return: The assigned class (0 or 1) and a score set to None for compliance reason.
    """
    res, res2 = visualize_substractions(inputfile=inputfile, plot=True)
    class_0_1 = 1
    if res < 0.045 and res2 < 0.14:
        class_0_1 = 0
    score = None
    return class_0_1, score

if __name__ == '__main__':
    df_global = build()
    print(df_global[(df_global['y_pred'] == 0) & (df_global['y_true'] == 1)])
    print(classifier_a_step2(inputfile='assets/anomalous_dice/img_17584_cropped.jpg'))
