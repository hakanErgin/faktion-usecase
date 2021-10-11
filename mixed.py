import matplotlib.pyplot as plt
import numpy as np
# import os
# import PIL
import tensorflow as tf

# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import pathlib
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


# I train with 15 epochs. Takes 4 sec for an epoch => 1 min total time
# but five epochs are also fine for testing.
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


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


# image_name = "358.jpg"
# image_path = "assets/optionB/3up_abnormal/"

# images from a custom Validation folder
# image_path = "assets/Validate/"
# image_path = "assets/optionB/5_abnormal/"
image_path = "assets/optionB_mixed/ready/Defected/"

# it will stop with an error at the last file
for i in range(360):
    try:
        image_name = f"Image{i}.jpg"
        # image_name = f"ab_{i}.jpg"

        img = tf.keras.utils.load_img(
            image_path + image_name, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "image {} most likely belongs to {} with a {:.2f} percent confidence."
            .format(i, class_names[np.argmax(score)], 100 * np.max(score)))
    except FileNotFoundError:
        print('end of files of file is missing')

    # print(score)

    # if scores for classification are not high, uncomment below to see what classes are confusing for the model
    # code below does nothing useful if you have only 2 classes, keep it commented

    # plt.figure(figsize=(8, 8))
    # plt.scatter(score*100, class_names)
    # plt.title(f'Probability classes for image {i}')
    # plt.xlabel('probability, %');
