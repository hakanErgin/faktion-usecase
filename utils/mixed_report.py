import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import pathlib

def build_model():
    # dataset_dir = "assets/optionB"
    dataset_dir = "assets/optionB_mixed/ready"
    data_dir = pathlib.Path(dataset_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 32
    img_height = 128
    img_width = 128

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    num_classes = 2

    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # UPD: that was only true for a small original dataset
    # I train with 15 epochs. Takes 4 sec for an epoch => 1 min total time
    # but five epochs are also fine for testing.
    epochs = 1
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # you may uncomment it to see the plot of training and validation accuracy
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    """
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
    model, img_height, img_width, class_names = build_model()
    df_normal = classifier_b(model, img_height, img_width, class_names, path='./assets/normal_dice/0/')
    for i in range(1, 11):
        df_normal = pd.DataFrame.append(df_normal, classifier_b(model, img_height, img_width, class_names, path='./assets/normal_dice/'+str(i)+'/'))

    df_normal['y_true'] = 0
    df_normal['y_pred'] = np.where(df_normal['result'] == 'Normal', 0, 1)

    df_anomalous = classifier_b(model, img_height, img_width, class_names, path='./assets/anomalous_dice/')
    df_anomalous['y_true'] = 1
    df_anomalous['y_pred'] = np.where(df_anomalous['result'] == 'Defected', 1, 0)

    df_global = pd.concat([df_normal, df_anomalous])
    # print(df_global.shape)
    # print(df_global.head(10).to_markdown())

    report = metrics.classification_report(df_global['y_true'], df_global['y_pred'])
    confus_matrix = metrics.confusion_matrix(df_global['y_true'], df_global['y_pred'])
    print(report)
    print(confus_matrix)
