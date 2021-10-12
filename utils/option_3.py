import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib



def get_model():
    model = keras.models.load_model('../cnn.pickled')
    dataset_dir = "assets/optionB_mixed/ready"
    data_dir = pathlib.Path(dataset_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 32
    img_height = 128
    img_width = 128

    class_names = train_ds.class_names
    print(class_names)
    return model, img_height, img_width, class_names

# ============================= #
# Classification report. Done by Lyes #


def classifier_b(model, img_height, img_width, class_names, path):
    df = pd.DataFrame(columns=["id", "result"])
    ids = []
    datas = []

    for filename in os.listdir(path):
        id = filename[:-4]
        img = tf.keras.utils.load_img(
            path + filename, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        res = class_names[np.argmax(score)]
        ids.append(id)
        datas.append(res)
        dfrow = [id]
        dfrow.append(res)
        df.loc[len(df)] = dfrow
    return df

def report():
    model, img_height, img_width, class_names = get_model()
    df_normal = classifier_b(model, img_height, img_width, class_names, path='./assets/normal_dice/0/')
    for i in range(1, 11):
        df_normal = pd.DataFrame.append(df_normal, classifier_b(model, img_height, img_width, class_names, path='./assets/normal_dice/'+str(i)+'/'))

    df_normal['y_true'] = 0
    df_normal['y_pred'] = np.where(df_normal['result'] == 'Normal', 0, 1)

    df_anomalous = classifier_b(model, img_height, img_width, class_names, path='./assets/anomalous_dice/')
    df_anomalous['y_true'] = 1
    df_anomalous['y_pred'] = np.where(df_anomalous['result'] == 'Defected', 1, 0)

    df_global = pd.concat([df_normal, df_anomalous])

    report = metrics.classification_report(df_global['y_true'], df_global['y_pred'])
    confus_matrix = metrics.confusion_matrix(df_global['y_true'], df_global['y_pred'])
    print(report)
    print(confus_matrix)
