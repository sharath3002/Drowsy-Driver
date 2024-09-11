# 1. **Environment Setup and Data Preparation**

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pathlib

data_dir = pathlib.Path('ArecanutDiseases')
print(data_dir)

# 2. **Loading and Preprocessing the Dataset**

img_height, img_width = 256, 256
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# 3. **Exploring the Dataset**

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(7):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i].numpy().argmax()])
    plt.axis("off")
plt.show()

# 4. **Building the Model with Transfer Learning**

resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(256, 256, 3),
    pooling='avg',
    classes=7,
    weights='imagenet'
)

for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(7, activation='softmax'))

resnet_model.summary()

# 5. **Compiling the Model**

resnet_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. **Inspecting a Batch of Data**

for images, labels in train_ds.take(1):
    print("Batch of training data shape:", images.shape)
    print("Batch of training labels shape:", labels.shape)

for images, labels in val_ds.take(1):
    print("Batch of validation data shape:", images.shape)
    print("Batch of validation labels shape:", labels.shape)

# 7. **Training the Model**

epochs = 10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 8. **Plotting Training Results**

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])

plt.show()

# 9. **Making Predictions**

import cv2

Anthracnose = list(data_dir.glob('Anthracnose/*'))
image_path = str(Anthracnose[0])
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)
print(image.shape)

pred = resnet_model.predict(image)
print(pred)

output_class = class_names[np.argmax(pred)]
print("The predicted class is", output_class)

# 10. **Saving the Model**

import os

current_dir = os.getcwd()
model_filename = "transferlearning_model.h5"
model_path = os.path.join(current_dir, model_filename)

resnet_model.save(model_path)
print("Model saved successfully at:", model_path)

# 11. **Evaluating the Model**

loss, accuracy = resnet_model.evaluate(val_ds)
print("Validation Accuracy:", accuracy)
